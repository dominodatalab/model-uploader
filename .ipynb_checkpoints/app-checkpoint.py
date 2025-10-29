# app.py
from __future__ import annotations

import os
import re
import io
import json
import time
import shutil
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, request, jsonify
import subprocess
import os
import requests
import subprocess
import tempfile
import shutil
import json
import time
from pathlib import Path
from urllib.parse import urljoin
from flask import Flask, render_template, request, Response, jsonify
import logging

app = Flask(__name__, static_url_path='/static')

# Balanced logging - keep useful info, reduce noise
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Keep werkzeug for request logs, reduce urllib3 debug spam
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

DOMINO_DOMAIN = os.environ.get("DOMINO_DOMAIN", "")
DOMINO_API_KEY = os.environ.get("DOMINO_API_KEY", "")

logger.info(f"DOMINO_DOMAIN: {DOMINO_DOMAIN}")
logger.info(f"DOMINO_API_KEY: {'***' if DOMINO_API_KEY else 'NOT SET'}")

DEFAULT_FILE_REGEX = r"\.py$"  # only scan Python files by default
MAX_WORKERS = int(os.environ.get("SEC_SCAN_MAX_WORKERS", "16"))
DEFAULT_SEMGREP_CONFIG = os.environ.get("SEMGREP_CONFIG", "p/default")

# ─────────────────────────────── HTTP Helpers ────────────────────────────────
class DominoApiError(RuntimeError):
    pass

class DominoClient:
    def __init__(self, base: str, api_key: str, timeout: int = 30):
        if not base or not api_key:
            raise DominoApiError("Missing DOMINO_DOMAIN or DOMINO_API_KEY")
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({
            "X-Domino-Api-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "domino-secscan/1.0",
        })

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def get_json(self, path: str, params: Optional[dict] = None) -> dict:
        url = self._url(path)
        r = self.s.get(url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            raise DominoApiError(f"GET {url} -> {r.status_code} {r.text[:300]}")
        try:
            return r.json()
        except Exception:
            raise DominoApiError(f"Non-JSON response from {url}")

    def get_bytes(self, path: str, params: Optional[dict] = None) -> bytes:
        url = self._url(path)
        # Override Accept to allow raw content
        headers = {**self.s.headers, "Accept": "*/*"}
        r = self.s.get(url, params=params, headers=headers, timeout=self.timeout)
        if r.status_code != 200:
            raise DominoApiError(f"GET {url} -> {r.status_code} ({r.headers.get('content-type')})")
        return r.content

# ───────────────────────────── Domino API Calls ─────────────────────────────

def get_registered_model_version(dc: DominoClient, model_name: str, version: int) -> dict:
    return dc.get_json(f"/api/registeredmodels/v1/{requests.utils.quote(model_name)}/versions/{version}")


def get_git_browse(dc: DominoClient, owner_username: str, project_name: str) -> dict:
    return dc.get_json("/v4/code/gitBrowse", params={
        "ownerUsername": owner_username,
        "projectName": project_name,
    })


def list_repo_paths(
    dc: DominoClient,
    project_id: str,
    repo_id: str,
    commit: str,
    include_regex: Optional[re.Pattern] = None,
    exclude_regex: Optional[re.Pattern] = None,
    max_files: int = 10000,
) -> List[str]:
    """
    Recursively list repo files at a commit using /git/browse.

    - Do NOT descend into directories that match exclude_regex.
    - Skip files that match exclude_regex.
    - Keep files that match include_regex (or everything if include_regex is None).
    """
    paths: List[str] = []
    stack: List[str] = [""]  # "" = repo root

    while stack:
        directory = stack.pop()
        params = {"commit": commit}
        if directory:
            params["directory"] = directory

        payload = dc.get_json(
            f"/v4/projects/{project_id}/gitRepositories/{repo_id}/git/browse",
            params,
        )
        items = (payload or {}).get("data", {}).get("items", [])
        for it in items:
            kind = it.get("kind")
            path = it.get("path") or it.get("name")
            if not path:
                continue

            if kind == "dir":
                # add trailing slash so exclude patterns like .../dir/ match directories only
                dir_key = path + "/"
                if exclude_regex and exclude_regex.search(dir_key):
                    continue  # block descent
                stack.append(path)

            elif kind == "file":
                if exclude_regex and exclude_regex.search(path):
                    continue
                if include_regex is None or include_regex.search(path):
                    paths.append(path)
                    if len(paths) >= max_files:
                        logger.warning("Reached max_files cap: %d", max_files)
                        return paths

    return sorted(paths)


def fetch_file_bytes(dc: DominoClient, project_id: str, repo_id: str, commit: str, path: str) -> bytes:
    return dc.get_bytes(f"/v4/projects/{project_id}/gitRepositories/{repo_id}/git/raw",
                        params={"fileName": path, "commit": commit})

# ───────────────────────────── Repo Materialization ──────────────────────────

def materialize_repo(
    dc: DominoClient,
    project_id: str,
    repo_id: str,
    commit: str,
    file_regex: Optional[str],
    exclude_regex: Optional[str],
    max_files: int,
    workers: int = MAX_WORKERS,
) -> Tuple[str, List[str]]:
    """Creates a temp dir, downloads matching files at the commit, returns (dir, paths)."""
    # include: None/".*" means ALL files
    include_re = None
    if file_regex and file_regex not in (".*", "*", "ALL"):
        include_re = re.compile(file_regex)

    # exclude: compile if provided
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    repo_dir = tempfile.mkdtemp(prefix="domino_repo_")

    # List the file paths first (now with include/exclude applied)
    paths = list_repo_paths(
        dc, project_id, repo_id, commit, include_re, exclude_re, max_files
    )
    if not paths:
        return repo_dir, []

    errors: List[str] = []

    def _download_and_write(p: str) -> Optional[str]:
        try:
            content = fetch_file_bytes(dc, project_id, repo_id, commit, p)
            abs_path = Path(repo_dir, p)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_path, "wb") as f:
                f.write(content)
            return p
        except Exception as e:
            errors.append(f"{p}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_and_write, p) for p in paths]
        for _ in as_completed(futs):
            pass

    if errors:
        logger.warning("Some files failed to fetch: %s", errors[:5])

    written = [p for p in paths if Path(repo_dir, p).exists()]
    return repo_dir, written

# ───────────────────────────── Semgrep Integration ───────────────────────────

def check_semgrep() -> Tuple[bool, Optional[str]]:
    try:
        r = subprocess.run(["semgrep", "--version"], capture_output=True, text=True)
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, r.stdout or r.stderr
    except FileNotFoundError:
        return False, "semgrep not found in PATH"


def run_semgrep_scan(target_dir: str, config: str = DEFAULT_SEMGREP_CONFIG, timeout_sec: int = 300) -> dict:
    ok, msg = check_semgrep()
    if not ok:
        raise RuntimeError(f"Semgrep not available: {msg}")

    cmd = [
        "semgrep", "--config", config,
        "--json",
        "--no-git-ignore",
        "--exclude", "*/tests/*",
        "--exclude", "*/test*/*", 
        "--exclude", "*/.git/*",
        "--exclude", "*/venv/*",
        "--exclude", "*/env/*",
        "--exclude", "*/__pycache__/*",
        target_dir
    ]
    
    logger.info(f"Running semgrep command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    
    logger.info(f"Semgrep exit code: {proc.returncode}")
    logger.info(f"Semgrep stdout length: {len(proc.stdout or '')}")
    logger.info(f"Semgrep stderr length: {len(proc.stderr or '')}")
    
    if proc.stderr:
        logger.warning(f"Semgrep stderr: {proc.stderr[:500]}")

    # semgrep exits 0 when no issues, 1 when issues found, >1 for errors
    if proc.returncode in (0, 1):
        try:
            result = json.loads(proc.stdout or '{"results": []}')
            logger.info(f"Semgrep found {len(result.get('results', []))} issues")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse semgrep JSON: {e}")
            logger.error(f"Raw stdout: {proc.stdout[:1000]}")
            raise RuntimeError(f"Failed to parse semgrep JSON: {e}\nSTDOUT[:500]: {proc.stdout[:500]}")
    
    # For non-zero/non-one exit codes, provide more detailed error info
    error_msg = f"Semgrep failed (code {proc.returncode})"
    if proc.stderr:
        error_msg += f": {proc.stderr[:300]}"
    if proc.stdout:
        error_msg += f" | stdout: {proc.stdout[:300]}"
    
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def summarize_semgrep(output: dict) -> dict:
    results = output.get("results", []) if isinstance(output, dict) else []
    sev = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    issues = []
    for r in results:
        # Map semgrep severity to bandit-style levels
        semgrep_sev = r.get("extra", {}).get("severity", "INFO").upper()
        # Convert semgrep severities to bandit-style
        if semgrep_sev == "ERROR":
            mapped_sev = "HIGH"
        elif semgrep_sev == "WARNING":
            mapped_sev = "MEDIUM"
        elif semgrep_sev == "INFO":
            mapped_sev = "LOW"
        else:
            mapped_sev = "LOW"
            
        if mapped_sev in sev:
            sev[mapped_sev] += 1
        
        issues.append({
            "filename": r.get("path"),
            "line_number": r.get("start", {}).get("line"),
            "test_id": r.get("check_id"),
            "test_name": r.get("extra", {}).get("message", ""),
            "issue_severity": mapped_sev,
            "issue_confidence": "HIGH",  # semgrep doesn't have confidence levels
            "issue_text": r.get("extra", {}).get("message", ""),
        })
    return {
        "total_issues": len(results),
        "high": sev["HIGH"],
        "medium": sev["MEDIUM"],
        "low": sev["LOW"],
        "issues": issues,
        "metrics": output.get("paths", {}),
    }

# ───────────────────────────── HTTP Endpoint ────────────────────────────────
@app.route("/security-scan-model", methods=["POST"])
def security_scan_model():
    t0 = time.time()
    try:
        body = request.get_json(silent=True) or {}
        model_name = body.get("modelName")
        version = body.get("version")
        include_issues = bool(body.get("includeIssues", True))
        include_metrics = bool(body.get("includeMetrics", False))
        file_regex = body.get("fileRegex", DEFAULT_FILE_REGEX)
        exclude_regex = body.get(
            "excludeRegex",
            r"(^|/)(node_modules|\.git|\.venv|\.streamlit|venv|env|__pycache__|\.ipynb_checkpoints)(/|$)"
        )

        max_files = int(body.get("maxFiles", 5000))
        timeout_sec = int(body.get("timeoutSec", 300))
        semgrep_config = body.get("semgrepConfig", DEFAULT_SEMGREP_CONFIG)

        if not model_name or version is None:
            return jsonify({"error": "modelName and version are required"}), 400

        dc = DominoClient(DOMINO_DOMAIN, DOMINO_API_KEY)

        # 1) Registered model version → commit, experimentRunId, project info
        mv = get_registered_model_version(dc, model_name, int(version))
        tags = mv.get("tags", {}) or {}
        commit = tags.get("mlflow.source.git.commit")
        owner_username = mv.get("ownerUsername") or mv.get("project", {}).get("ownerUsername")
        project_id = mv.get("project", {}).get("id") or tags.get("mlflow.domino.project_id")
        project_name = mv.get("project", {}).get("name") or tags.get("mlflow.domino.project_name")
        run_url_rel = mv.get("versionUiDetails", {}).get("experimentRunInfo", {}).get("runUrl")
        run_url = f"{DOMINO_DOMAIN}{run_url_rel}" if run_url_rel else None
        experiment_run_id = mv.get("experimentRunId")

        if not (owner_username and project_name and project_id):
            return jsonify({"error": "Unable to resolve ownerUsername/projectName/projectId from model"}), 500
        if not commit:
            return jsonify({"error": "Model version missing tags.mlflow.source.git.commit; cannot pin snapshot."}), 400

        # 2) Resolve main repository id/uri via gitBrowse
        gb = get_git_browse(dc, owner_username, project_name)
        repo_id = gb.get("projectMainRepositoryId")
        repo_uri = gb.get("projectMainRepositoryUri")
        if not repo_id:
            return jsonify({"error": "No main repository found for project"}), 404

        # 3) Download repo at commit to temp dir (only files matching regex)
        repo_dir, file_paths = materialize_repo(dc, project_id, repo_id, commit, file_regex, exclude_regex, max_files)
        if not file_paths:
            shutil.rmtree(repo_dir, ignore_errors=True)
            return jsonify({"error": "No files to scan after filtering", "regex": file_regex, "excludeRegex": exclude_regex}), 404

        # 4) Semgrep scan
        try:
            logger.info(f"Starting semgrep scan on {len(file_paths)} files in {repo_dir}")
            logger.info(f"Using semgrep config: {semgrep_config}")
            semgrep_raw = run_semgrep_scan(repo_dir, config=semgrep_config, timeout_sec=timeout_sec)
        finally:
            shutil.rmtree(repo_dir, ignore_errors=True)

        summary = summarize_semgrep(semgrep_raw)

        result = {
            "summary": summary,
            "model": {
                "modelName": mv.get("modelName"),
                "modelVersion": mv.get("modelVersion"),
                "experimentRunId": experiment_run_id,
                "runUrl": run_url,
                "project": {"id": project_id, "name": project_name},
                "git": {
                    "commit": commit,
                    "projectMainRepositoryId": repo_id,
                    "projectMainRepositoryUri": repo_uri,
                },
            },
            "scan": {
                "total": summary["total_issues"],
                "high": summary["high"],
                "medium": summary["medium"],
                "low": summary["low"],
                "file_count_scanned": len(file_paths),
                "file_regex": file_regex,
                "exclude_regex": exclude_regex,
                "duration_sec": round(time.time() - t0, 3),
            },
        }
        if include_issues:
            result["issues"] = summary["issues"]
        if include_metrics:
            result["metrics"] = summary.get("metrics")

        return jsonify(result)

    except DominoApiError as e:
        logger.exception("Domino API error")
        return jsonify({"error": str(e)}), 502
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Semgrep timed out"}), 504
    except Exception as e:
        logger.exception("Unexpected error in security_scan_model")
        return jsonify({"error": f"Unexpected error: {e}"}), 500


def make_domino_api_request(endpoint, method='GET'):
    """Make authenticated request to Domino API"""
    url = f"{DOMINO_DOMAIN}/{endpoint.lstrip('/')}"
    headers = {
        'X-Domino-Api-Key': DOMINO_API_KEY,
        'Accept': 'application/json'
    }
    
    try:
        response = requests.request(method, url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Domino API request failed: {e}")
        raise

def get_experiment_run_details(experiment_id):
    """Get experiment run details including repository info"""
    try:
        # First try to get run details directly
        run_data = make_domino_api_request(f'api/runs/v1/runs/{experiment_id}')
        return run_data
    except Exception as e:
        logger.error(f"Failed to get run details for {experiment_id}: {e}")
        raise


# Test curl command at startup
def test_api_connectivity():
    if not DOMINO_DOMAIN or not DOMINO_API_KEY:
        logger.error("Missing DOMINO_DOMAIN or DOMINO_API_KEY environment variables")
        return
    
    test_url = f"{DOMINO_DOMAIN}/api/governance/v1/bundles"
    
    # Build the exact curl command
    curl_cmd_str = f"curl -s -w 'HTTP_CODE:%{{http_code}}' -H 'X-Domino-Api-Key: {DOMINO_API_KEY}' -H 'Accept: application/json' '{test_url}'"
    curl_cmd = [
        'curl', '-s', '-w', 'HTTP_CODE:%{http_code}',
        '-H', f'X-Domino-Api-Key: {DOMINO_API_KEY}',
        '-H', 'Accept: application/json',
        test_url
    ]
    
    try:
        logger.info(f"Testing API connectivity with:")
        logger.info(f"  {curl_cmd_str}")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        logger.info(f"Curl exit code: {result.returncode}")
        logger.info(f"Curl stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Curl stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Curl command timed out after 30 seconds")
        logger.info(f"Copy/paste to test manually: {curl_cmd_str}")
    except Exception as e:
        logger.error(f"Curl command failed: {str(e)}")
        logger.info(f"Copy/paste to test manually: {curl_cmd_str}")

# Health check endpoints
@app.route("/_stcore/health")
def health():
    return "", 200

@app.route("/_stcore/host-config")
def host_config():
    return "", 200

@app.route("/proxy/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy_request(path):
    logger.info(f"Proxy request: {request.method} {path}")
    
    if request.method == "OPTIONS":
        return "", 204
    
    # Get target URL from query param
    target_base = request.args.get('target')
    if not target_base:
        error_msg = "Missing target URL. Use ?target=https://api.example.com"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 400
    
    # Build upstream URL
    upstream_url = urljoin(target_base.rstrip("/") + "/", path)
    
    # Forward headers (exclude hop-by-hop headers and conflicting auth)
    forward_headers = {}
    skip_headers = {
        "host", "content-length", "transfer-encoding", "connection", "keep-alive",
        "authorization"  # Skip this - conflicts with X-Domino-Api-Key
    }
    
    for key, value in request.headers:
        if key.lower() not in skip_headers:
            forward_headers[key] = value
    
    # Filter out the 'target' parameter from upstream request
    upstream_params = {k: v for k, v in request.args.items() if k != 'target'}
    
    logger.info(f"Making upstream request: {request.method} {upstream_url}")
    if upstream_params:
        logger.info(f"Upstream params: {upstream_params}")
    
    # Log the equivalent curl command for debugging
    headers_str = " ".join([f"-H '{k}: {v}'" for k, v in forward_headers.items()])
    params_str = "&".join([f"{k}={v}" for k, v in upstream_params.items()])
    final_url = f"{upstream_url}?{params_str}" if params_str else upstream_url
    curl_equivalent = f"curl -X {request.method} {headers_str} '{final_url}'"
    logger.info(f"Equivalent curl command:")
    logger.info(f"  {curl_equivalent}")
    
    try:
        # Make the upstream request
        resp = requests.request(
            method=request.method,
            url=upstream_url,
            params=upstream_params,
            data=request.get_data(),
            headers=forward_headers,
            timeout=30,
            stream=True
        )
        
        logger.info(f"Upstream response: {resp.status_code}")
        
        # Log response body for debugging (truncated)
        if resp.status_code >= 400:
            try:
                # Get a copy of the content for logging
                content = resp.content
                logger.error(f"Upstream error response body: {content[:1000].decode('utf-8', errors='ignore')}")
                # Create new response with the content
                response_headers = []
                hop_by_hop = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
                
                for key, value in resp.headers.items():
                    if key.lower() not in hop_by_hop:
                        response_headers.append((key, value))
                
                return Response(
                    content,
                    status=resp.status_code,
                    headers=response_headers
                )
            except Exception as e:
                logger.error(f"Error reading response content: {str(e)}")
        
        # Forward response headers (exclude hop-by-hop)
        response_headers = []
        hop_by_hop = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
        
        for key, value in resp.headers.items():
            if key.lower() not in hop_by_hop:
                response_headers.append((key, value))
        
        return Response(
            resp.iter_content(chunk_size=8192),
            status=resp.status_code,
            headers=response_headers,
            direct_passthrough=True
        )
        
    except requests.RequestException as e:
        error_msg = f"Proxy request failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 502
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

# Page routes
def safe_domino_config():
    return {
        "PROJECT_ID": os.environ.get("DOMINO_PROJECT_ID", ""),
        "RUN_HOST_PATH": os.environ.get("DOMINO_RUN_HOST_PATH", ""),
        "API_BASE": DOMINO_DOMAIN,
        "API_KEY": DOMINO_API_KEY,   
    }

@app.route("/")
def home():
    return render_template("index.html", DOMINO=safe_domino_config())

@app.route("/original")
def original():
    return render_template("original_index.html", DOMINO=safe_domino_config())

if __name__ == "__main__":
    # Test API connectivity on startup
    test_api_connectivity()
    
    port = int(os.environ.get("PORT", 8888))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)