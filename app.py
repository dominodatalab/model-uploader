import os
import logging
from urllib.parse import urljoin

import requests
from flask import Flask, render_template, request, Response, jsonify
import queue
import json

from model_registration import register_model_handler

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
DOMINO_PROJECT_ID = os.environ.get("DOMINO_PROJECT_ID", "")

logger.info(f"DOMINO_DOMAIN: {DOMINO_DOMAIN}")
logger.info(f"DOMINO_API_KEY: {'***' if DOMINO_API_KEY else 'NOT SET'}")
logger.info(f"DOMINO_PROJECT_ID: {DOMINO_PROJECT_ID}")

progress_queues = {}


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
    return register_model_handler(request, progress_queues)


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