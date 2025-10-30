import os
import hashlib
import base64
import uuid
import time
import shutil
import logging
import tempfile
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
import mlflow
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__, static_url_path='/static')

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


def save_uploaded_files(files, temp_dir):
    """Save uploaded files to temp directory maintaining structure."""
    saved_files = []
    for file in files:
        filepath = Path(temp_dir) / file.filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(filepath))
        saved_files.append(str(filepath))
    return saved_files


def log_metadata_as_metrics(metadata_path: str):
    """Load metadata JSON and log all numeric values as MLflow metrics."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary into dot-separated keys."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_metadata = flatten_dict(metadata)
        
        for key, value in flat_metadata.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)
                logger.info(f"Logged metric: {key} = {value}")
            elif isinstance(value, bool):
                mlflow.log_metric(key, int(value))
                logger.info(f"Logged metric: {key} = {int(value)}")
            else:
                mlflow.log_param(key, str(value))
                logger.info(f"Logged param: {key} = {value}")
                
    except Exception as e:
        logger.warning(f"Could not log metadata as metrics: {e}")


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


@app.route("/_stcore/health")
def health():
    return "", 200


@app.route("/_stcore/host-config")
def host_config():
    return "", 200


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
        
        # Save uploaded files
        files = request.files.getlist('files')
        temp_dir = tempfile.mkdtemp(prefix=f"external_model_{model_name}_")
        logger.info(f"Created temp directory: {temp_dir}")
        
        saved_files = save_uploaded_files(files, temp_dir)
        logger.info(f"Saved {len(saved_files)} files to temp directory")
        
        # Find required files
        model_pkl = None
        inference_py = None
        metadata_json = None
        requirements_txt = None
        
        for filepath in saved_files:
            filename = Path(filepath).name
            if filename == "model.pkl":
                model_pkl = filepath
            elif filename == "inference.py":
                inference_py = filepath
            elif filename == "metadata.json":
                metadata_json = filepath
            elif filename == "requirements.txt":
                requirements_txt = filepath
        
        # Register with MLflow
        exp = mlflow.set_experiment(EXPERIMENT_NAME)
        experiment_id = exp.experiment_id
        logger.info(f"Using MLflow experiment: {EXPERIMENT_NAME}")
        
        with mlflow.start_run(run_name=f"{model_name}_registration_{int(time.time())}") as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Log parameters
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
            
            # Log artifacts
            for filepath in saved_files:
                rel_path = Path(filepath).relative_to(temp_dir)
                artifact_dir = str(rel_path.parent) if rel_path.parent != Path(".") else None
                mlflow.log_artifact(filepath, artifact_path=artifact_dir)
                logger.info(f"Logged artifact: {rel_path}")
            
            # Log metadata as metrics if available
            if metadata_json:
                log_metadata_as_metrics(metadata_json)
            
            # Log model
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
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment: {EXPERIMENT_NAME}")
        
        # Update model description via Domino API
        model_data = update_model_description(model_name, model_description)
        model_version = model_data.get("latestVersion", 1)
        
        # Create bundle
        policy_id = POLICY_IDS_LIST[0] if POLICY_IDS_LIST else ""
        bundle_data = create_bundle(model_name, model_version, policy_id)
        
        # Extract project owner and name from bundle_data
        project_owner = bundle_data.get("projectOwner", "")
        project_name = bundle_data.get("projectName", "")
        bundle_id = bundle_data.get("id", "")
        stage = bundle_data.get("stage", "").lower().replace(" ", "-")
        
        # Construct URLs
        domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
        experiment_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}"
        experiment_run_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}/{run_id}"
        model_artifacts_url = f"https://{domain}/experiments/{project_owner}/{project_name}/{experiment_id}/{run_id}?isdir=false&path=model%2FMLmodel&tab=Outputs"
        model_card_url = f"https://{domain}/u/{project_owner}/{project_name}/model-registry/{model_name}/model-card?version={model_version}"
        bundle_url = f"https://{domain}/u/{project_owner}/{project_name}/governance/bundle/{bundle_id}/policy/{policy_id}/evidence/stage/{stage}"

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
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error registering model: {e}", exc_info=True)
        
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        
        return jsonify({
            "status": "error",
            "message": f"Failed to register model: {str(e)}"
        }), 500


def safe_domino_config():
    """Return sanitized Domino configuration for templates."""
    return {
        "PROJECT_ID": os.environ.get("DOMINO_PROJECT_ID", ""),
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