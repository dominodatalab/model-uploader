# endpoint_registration.py
import os
import logging
import requests

logger = logging.getLogger(__name__)

DOMINO_DOMAIN = os.environ.get("DOMINO_DOMAIN", "se-demo.domino.tech")
DOMINO_API_KEY = os.environ.get("DOMINO_USER_API_KEY", "")
DOMINO_PROJECT_ID = os.environ.get("DOMINO_PROJECT_ID", "")


def register_endpoint(bundle_id: str, bundle_name: str, model_name: str, 
                     model_version: int, training_set_id: str = None) -> dict:
    """Register a Model API endpoint via Domino API."""
    domain = DOMINO_DOMAIN.removeprefix("https://").removeprefix("http://")
    url = f"https://{domain}/api/modelServing/v1/modelApis"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Domino-Api-Key": DOMINO_API_KEY
    }
    
    payload = {
        "bundleId": bundle_id,
        "description": f"Auto-generated endpoint for model {model_name} version {model_version}",
        "environmentId": "6597567f09b1d67423f20b08",
        "environmentVariables": [
            {
                "key": "username",
                "value": "myself"
            }
        ],
        "isAsync": False,
        "name": bundle_name,
        "strictNodeAntiAffinity": False,
        "version": {
            "logHttpRequestResponse": True,
            "monitoringEnabled": True,
            "projectId": DOMINO_PROJECT_ID,
            "shouldDeploy": True,
            "predictionDatasetResourceId": training_set_id,
            "source": {
                "registeredModelName": model_name,
                "registeredModelVersion": model_version,
                "type": "Registry"
            }
        }
    }
    
    try:
        logger.info(f"Registering endpoint for model {model_name} v{model_version}")
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        endpoint_data = response.json()
        logger.info(f"Successfully created Model API endpoint: {endpoint_data.get('id')}")
        return endpoint_data
    except requests.RequestException as e:
        logger.error(f"Failed to register endpoint: {e}")
        raise