import os, json, requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST", "https://dbc-496010ba-2046.cloud.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
ENDPOINT_URL     = f"{DATABRICKS_HOST}/serving-endpoints/reai-combined/invocations"


def call_endpoint(model_type: str, payload: list):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type":  "application/json",
    }
    body = {
        "dataframe_records": [{
            "model_type": model_type,
            "payload":    json.dumps(payload),
        }]
    }
    resp = requests.post(ENDPOINT_URL, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    # MLflow pyfunc returns {"predictions": [{"result": "<json string>"}]}
    raw = resp.json()["predictions"][0]["result"]
    return json.loads(raw)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/avm", methods=["POST"])
def avm():
    try:
        data   = request.json
        result = call_endpoint("avm", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/lead", methods=["POST"])
def lead():
    try:
        data   = request.json
        result = call_endpoint("lead_scoring", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        data   = request.json
        result = call_endpoint("price_forecast", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
