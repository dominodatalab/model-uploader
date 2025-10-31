import os
import hashlib
import base64
import uuid
import time
import shutil
import logging
import tempfile
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
import mlflow
from flask import Flask, render_template, request, Response, jsonify
import queue
import threading

app = Flask(__name__, static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

DOMINO_DOMAIN = os.environ.get("DOMINO_DOMAIN", "se-demo.domino.tech")
DOMINO_API_KEY = os.environ.get("DOMINO_USER_API_KEY", "")
POLICY_IDS_STRING = os.environ.get("POLICY_IDS", "42c9adf3-f233-470b-b186-107496d0eb05, ")
POLICY_IDS_LIST = [s.strip() for s in POLICY_IDS_STRING.split(',')]
DOMINO_PROJECT_ID = os.environ.get("DOMINO_PROJECT_ID", "")

logger.info(f"DOMINO_DOMAIN: {DOMINO_DOMAIN}")
logger.info(f"DOMINO_API_KEY: {'***' if DOMINO_API_KEY else 'NOT SET'}")
logger.info(f"DOMINO_PROJECT_ID: {DOMINO_PROJECT_ID}")
progress_queues = {}


def domino_short_id(length: int = 8) -> str:
    """Generate a short ID based on Domino user and project."""
    def short_fallback():
        return base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")[:length]

    user = os.environ.get("DOMINO_USER_NAME") or short_fallback()
    project = os.environ.get("DOMINO_PROJECT_ID") or short_fallback()
    combined = f"{user}/{project}"
    digest = hashlib.sha256(combined.encode()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return f"{user}_{encoded[:length]}"


EXPERIMENT_NAME = f"external_models_{domino_short_id(4)}"

def send_progress(request_id, step, message, progress=None, file_status=None):
    """Send progress update to the frontend."""
    if request_id in progress_queues:
        progress_queues[request_id].put({
            'step': step,
            'message': message,
            'progress': progress,
            'file_status': file_status
        })

def save_uploaded_files(files, temp_dir):
    """Save uploaded files to temp directory maintaining structure."""
    saved_files = []
    for file in files:
        filepath = Path(temp_dir) / file.filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(filepath))
        filesize = filepath.stat().st_size  # bytes
        saved_files.append({
            "path": str(filepath),
            "size_bytes": filesize,
            "size_mb": round(filesize / (1024 * 1024), 2)
        })
    return saved_files


def update_model_description(model_name: str, description: str) -> dict:
    """Update model description via Domino API and return full response."""
    domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
    url = f"https://{domain}/api/registeredmodels/v1/{model_name}"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Domino-Api-Key": DOMINO_API_KEY
    }
    payload = {
        "description": description,
        "discoverable": True
    }
    
    try:
        logger.info(f"Updating model description for {model_name}")
        response = requests.patch(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        model_data = response.json()
        logger.info(f"Successfully updated model description for {model_name}")
        return model_data
    except requests.RequestException as e:
        logger.error(f"Failed to update model description: {e}")
        raise


def normalize_label(label: str) -> str:
    """Transform label to match evidence variable names."""
    normalized = label.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized)
    return normalized


def get_policy_details(policy_id: str) -> dict:
    """Get policy details including classification artifact map."""
    domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
    url = f"https://{domain}/api/governance/v1/policies/{policy_id}"
    headers = {
        "accept": "application/json",
        "X-Domino-Api-Key": DOMINO_API_KEY
    }
    
    try:
        logger.info(f"Getting policy details for policy ID: {policy_id}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        policy_data = response.json()
        logger.info(f"Successfully retrieved policy: {policy_data.get('name')}")
        
        # Extract unique tuples
        tuples = []
        policy_id_from_response = policy_data.get("id")
        stages = policy_data.get("stages", [])
        
        for stage in stages:
            for evidence in stage.get("evidenceSet", []):
                evidence_id = evidence.get("id")
                for artifact in evidence.get("artifacts", []):
                    artifact_id = artifact.get("id")
                    label = artifact.get("details", {}).get("label")
                    input_type = artifact.get("details", {}).get("type")
                    if policy_id_from_response and evidence_id and artifact_id and label and input_type:
                        tuples.append((policy_id_from_response, evidence_id, artifact_id, label, input_type))
        
        # Remove duplicates
        unique_tuples = list(dict.fromkeys(tuples))
        
        print("Unique Policy Artifact Tuples:")
        for t in unique_tuples:
            print(t)
        
        return policy_data
    except requests.RequestException as e:
        logger.error(f"Failed to get policy details: {e}")
        raise


def create_bundle(model_name: str, model_version: int, policy_id: str) -> dict:
    """Create a bundle for the registered model."""
    domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
    url = f"https://{domain}/api/governance/v1/bundles"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Domino-Api-Key": DOMINO_API_KEY
    }
    
    bundle_name = f"{model_name}_v{model_version}"
    payload = {
        "attachments": [
            {
                "identifier": {
                    "name": model_name,
                    "version": model_version
                },
                "type": "ModelVersion"
            }
        ],
        "name": bundle_name,
        "policyId": policy_id,
        "projectId": DOMINO_PROJECT_ID
    }
    
    try:
        logger.info(f"Creating bundle: {bundle_name}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        bundle_data = response.json()
        logger.info(f"Successfully created bundle: {bundle_data.get('id')}")
        return bundle_data
    except requests.RequestException as e:
        logger.error(f"Failed to create bundle: {e}")
        raise


def submit_artifacts_to_policy(bundle_id: str, policy_id: str, matched_artifacts: list) -> dict:
    """Submit all artifacts to policy in a single batch call."""
    domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
    url = f"https://{domain}/api/governance/v1/rpc/submit-result-to-policy"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Domino-Api-Key": DOMINO_API_KEY
    }
    
    # Group artifacts by evidence_id to make multiple calls if needed
    evidence_groups = {}
    for artifact in matched_artifacts:
        if artifact['value'] is not None:  # Only submit non-None values
            evidence_id = artifact['evidence_id']
            if evidence_id not in evidence_groups:
                evidence_groups[evidence_id] = []
            evidence_groups[evidence_id].append(artifact)
    
    logger.info(f"Submitting {len(matched_artifacts)} artifacts across {len(evidence_groups)} evidence sets")
    
    results = []
    for evidence_id, artifacts in evidence_groups.items():
        # Build content dict for this evidence group
        content = {}
        for artifact in artifacts:
            artifact_id = artifact['artifact_id']
            value = artifact['value']
            
            # Convert value based on input type
            if artifact['input_type'] == 'radio':
                if isinstance(value, bool):
                    content[artifact_id] = "Yes" if value else "No"
                else:
                    content[artifact_id] = str(value)
            else:
                content[artifact_id] = str(value)
        
        payload = {
            "bundleId": bundle_id,
            "content": content,
            "evidenceId": evidence_id,
            "policyId": policy_id
        }
        
        try:
            logger.info(f"Submitting evidence group {evidence_id} with {len(content)} artifacts")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result_data = response.json()
            results.append(result_data)
            logger.info(f"Successfully submitted evidence group {evidence_id}")
        except requests.RequestException as e:
            logger.error(f"Failed to submit evidence group {evidence_id}: {e}")
            raise
    
    # Return the last result (or could aggregate all results)
    return results[-1] if results else {}


@app.route("/_stcore/health")
def health():
    return "", 200


@app.route("/_stcore/host-config")
def host_config():
    return "", 200


@app.route("/register-progress/<request_id>")
def register_progress(request_id):
    """SSE endpoint for progress updates."""
    def generate():
        q = queue.Queue()
        progress_queues[request_id] = q
        try:
            while True:
                data = q.get()
                if data.get('done'):
                    break
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            if request_id in progress_queues:
                del progress_queues[request_id]
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/proxy/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy_request(path):
    """Proxy requests to upstream services."""
    logger.info(f"Proxy request: {request.method} {path}")
    
    if request.method == "OPTIONS":
        return "", 204
    
    target_base = request.args.get('target')
    if not target_base:
        return jsonify({"error": "Missing target URL. Use ?target=https://api.example.com"}), 400
    
    upstream_url = urljoin(target_base.rstrip("/") + "/", path)
    
    skip_headers = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "authorization"}
    forward_headers = {k: v for k, v in request.headers if k.lower() not in skip_headers}
    upstream_params = {k: v for k, v in request.args.items() if k != 'target'}
    
    logger.info(f"Making upstream request: {request.method} {upstream_url}")
    
    try:
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
        
        hop_by_hop = {"content-encoding", "transfer-encoding", "connection", "keep-alive"}
        response_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in hop_by_hop]
        
        if resp.status_code >= 400:
            try:
                content = resp.content
                logger.error(f"Upstream error response: {content[:1000].decode('utf-8', errors='ignore')}")
                return Response(content, status=resp.status_code, headers=response_headers)
            except Exception as e:
                logger.error(f"Error reading response content: {e}")
        
        return Response(
            resp.iter_content(chunk_size=8192),
            status=resp.status_code,
            headers=response_headers,
            direct_passthrough=True
        )
        
    except requests.RequestException as e:
        logger.error(f"Proxy request failed: {e}")
        return jsonify({"error": f"Proxy request failed: {e}"}), 502
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/register-external-model", methods=["POST"])
def register_external_model():
    """Register an external model with Domino using MLflow."""
    logger.info("=" * 80)
    logger.info("REGISTER EXTERNAL MODEL - Request Received")
    logger.info("=" * 80)
    
    temp_dir = None
    request_id = request.form.get("requestId", str(uuid.uuid4()))
    
    try:
        # Extract and validate form data
        model_name = request.form.get("modelName")
        model_description = request.form.get("modelDescription", "")
        model_owner = request.form.get("modelOwner")
        model_use_case = request.form.get("modelUseCase")
        model_usage_pattern = request.form.get("modelUsagePattern")
        model_environment_id = request.form.get("modelEnvironmentId")
        model_execution_script = request.form.get("modelExecutionScript", "")
        
        required_fields = [model_name, model_owner, model_use_case, model_usage_pattern, model_environment_id]
        if not all(required_fields):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        # Get policy details
        policy_id = POLICY_IDS_LIST[0] if POLICY_IDS_LIST else ""
        send_progress(request_id, 'policy', 'Retrieving policy details...', progress=5)
        policy_data = get_policy_details(policy_id)
        
        # Save uploaded files
        files = request.files.getlist('files')
        temp_dir = tempfile.mkdtemp(prefix=f"external_model_{model_name}_")
        logger.info(f"Created temp directory: {temp_dir}")
        
        send_progress(request_id, 'upload', 'Saving uploaded files...', progress=10)
        
        saved_files = save_uploaded_files(files, temp_dir)
        logger.info(f"Saved {len(saved_files)} files to temp directory")
        
        # Find required files and send file status
        file_status = {}
        model_pkl = None
        inference_py = None
        metadata_json = None
        requirements_txt = None

        list_the_file_names_and_sizes = ''
        for saved_file in saved_files:
            filepath = saved_file.get('path', '')
            filesize = saved_file.get('size_mb', '')
            rel_path = str(Path(filepath).relative_to(temp_dir))
            file_status[rel_path] = 'uploaded'
            filename = Path(filepath).name
            
            list_the_file_names_and_sizes += f"{filename}  -  {filesize}MB\n"

            if filename == "model.pkl":
                model_pkl = filepath
            elif filename == "inference.py":
                inference_py = filepath
            elif filename == "metadata.json":
                metadata_json = filepath
            elif filename == "requirements.txt":
                requirements_txt = filepath
        
        send_progress(request_id, 'validate', 'Files uploaded successfully', progress=20, file_status=file_status)
        
        # Register with MLflow
        exp = mlflow.set_experiment(EXPERIMENT_NAME)
        experiment_id = exp.experiment_id
        logger.info(f"Using MLflow experiment: {EXPERIMENT_NAME}")
        
        send_progress(request_id, 'mlflow', 'Starting MLflow run...', progress=30)
        
        with mlflow.start_run(run_name=f"{model_name}_registration_{int(time.time())}") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Log parameters
            send_progress(request_id, 'params', 'Logging parameters...', progress=40)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("model_description", model_description)
            mlflow.log_param("model_owner", model_owner)
            mlflow.log_param("model_use_case", model_use_case)
            mlflow.log_param("model_usage_pattern", model_usage_pattern)
            mlflow.log_param("model_environment_id", model_environment_id)
            mlflow.log_param("model_execution_script", model_execution_script)
            mlflow.log_param("registration_time", time.time())
            mlflow.log_param("file_count", len(saved_files))
            
            # Log artifacts with progress
            send_progress(request_id, 'artifacts', 'Logging artifacts...', progress=50)
            for i, saved_file in enumerate(saved_files):
                filepath = saved_file["path"]
                rel_path = Path(filepath).relative_to(temp_dir)
                artifact_dir = str(rel_path.parent) if rel_path.parent != Path(".") else None
                mlflow.log_artifact(filepath, artifact_path=artifact_dir)
                logger.info(f"Logged artifact: {rel_path}")
                
                file_status[str(rel_path)] = 'logged'
                progress_val = 50 + (i + 1) / len(saved_files) * 15
                send_progress(request_id, 'artifacts', f'Logged {rel_path}', progress=progress_val, file_status=file_status)
                        
            # Log model
            send_progress(request_id, 'model', 'Registering model...', progress=67)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                code_paths=[inference_py],
                artifacts={
                    "model_file": model_pkl,
                    "metadata_file": metadata_json
                },
                pip_requirements=requirements_txt,
                python_model=mlflow.pyfunc.PythonModel(),
                registered_model_name=model_name
            )
            
            logger.info(f"Successfully logged model to MLflow")
        
        send_progress(request_id, 'api', 'Updating model description...', progress=75)
        model_data = update_model_description(model_name, model_description)
        model_version = model_data.get("latestVersion", 1)
        
        # Create bundle
        send_progress(request_id, 'bundle', 'Creating governance bundle...', progress=80)
        bundle_data = create_bundle(model_name, model_version, policy_id)
        print('bd')
        print(bundle_data)

        # Extract project info and construct URLs
        project_owner = bundle_data.get("projectOwner", "")
        project_name = bundle_data.get("projectName", "")
        project_id = bundle_data.get("projectId", "")
        policy_name = bundle_data.get("policyName", "")
        bundle_id = bundle_data.get("id", "")
        stage = bundle_data.get("stage", "").lower().replace(" ", "-")
        
        domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
        experiment_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}"
        experiment_run_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}/{run_id}"
        model_artifacts_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}/{run_id}?isdir=false&path=model%2FMLmodel&tab=Outputs"
        model_card_url = f"https://{domain}/u/{project_owner}/{project_name}/model-registry/{model_name}/model-card?version={model_version}"
        bundle_url = f"https://{domain}/u/{project_owner}/{project_name}/governance/bundle/{bundle_id}/policy/{policy_id}/evidence/stage/{stage}"

        were_all_the_files_uploaded = True
        
        evidence_variables = {
            'model_name': model_name,
            'model_description': model_description,
            'model_owner': model_owner,
            'model_use_case': model_use_case,
            'model_usage_pattern': model_usage_pattern,
            'model_environment_id': model_environment_id,
            'model_execution_script': model_execution_script,
            'list_the_file_names_and_sizes': list_the_file_names_and_sizes,
            'were_all_files_uploaded': were_all_the_files_uploaded
        }
        
        # Match artifacts with evidence variables
        stages = policy_data.get("stages", [])
        
        matched_artifacts = []
        seen_keys = set()
        
        for stage in stages:
            for evidence in stage.get("evidenceSet", []):
                evidence_id = evidence.get("id")
                for artifact in evidence.get("artifacts", []):
                    artifact_id = artifact.get("id")
                    label = artifact.get("details", {}).get("label")
                    input_type = artifact.get("details", {}).get("type")
                    
                    if policy_id and evidence_id and artifact_id and label and input_type:
                        unique_key = (policy_id, evidence_id, artifact_id, label, input_type)
                        
                        if unique_key not in seen_keys:
                            seen_keys.add(unique_key)
                            normalized_key = normalize_label(label)
                            value = evidence_variables.get(normalized_key)
                            
                            matched_artifacts.append({
                                'bundle_id': bundle_id,
                                'policy_id': policy_id,
                                'evidence_id': evidence_id,
                                'artifact_id': artifact_id,
                                'label': label,
                                'input_type': input_type,
                                'value': value
                            })
        
        print("\nMatched Policy Artifacts with Evidence Variables:")
        for artifact in matched_artifacts:
            print(artifact)

        # Submit artifacts to policy
        send_progress(request_id, 'evidence', 'Submitting evidence to policy...', progress=90)
        policy_submission_result = submit_artifacts_to_policy(bundle_id, policy_id, matched_artifacts)
        logger.info(f"Successfully submitted {len(matched_artifacts)} artifacts to policy")

        send_progress(request_id, 'complete', 'Registration complete!', progress=100)
        send_progress(request_id, 'done', '', progress=100)
        
        # Clean up
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        
        logger.info("=" * 80)
        
        return jsonify({
            "status": "success",
            "message": "Model registered and bundle created successfully",
            "data": {
                "model_name": model_name,
                "model_version": model_version,
                "run_id": run_id,
                "experiment_name": EXPERIMENT_NAME,
                "file_count": len(saved_files),
                "bundle_id": bundle_id,
                "bundle_name": bundle_data.get("name"),
                "experiment_url": experiment_url,
                "experiment_run_url": experiment_run_url,
                "model_artifacts_url": model_artifacts_url,
                "model_card_url": model_card_url,
                "bundle_url": bundle_url,
                "policy_name": policy_name,
                "policy_id": policy_id,
                "project_id": DOMINO_PROJECT_ID,
                "project_name": project_name,
                "artifacts_submitted": len([a for a in matched_artifacts if a['value'] is not None])
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error registering model: {e}", exc_info=True)
        send_progress(request_id, 'error', f'Error: {str(e)}', progress=0)
        send_progress(request_id, 'done', '', progress=0)
        
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        
        return jsonify({
            "status": "error",
            "message": f"Failed to register model: {str(e)}"
        }), 500


def safe_domino_config():
    """Return sanitized Domino configuration for templates."""
    return {
        "PROJECT_ID": DOMINO_PROJECT_ID,
        "RUN_HOST_PATH": os.environ.get("DOMINO_RUN_HOST_PATH", ""),
        "API_BASE": DOMINO_DOMAIN,
        "API_KEY": DOMINO_API_KEY,
    }


@app.route("/")
def home():
    return render_template("index.html", DOMINO=safe_domino_config())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)