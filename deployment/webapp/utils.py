import requests

API_URL = "http://3.238.140.91/predict"   # your EC2 public API


def call_api(payload):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
