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
    resp = requests.post(ENDPOINT_URL, headers=headers, json=body, timeout=55)
    resp.raise_for_status()

    # MLflow pyfunc returns {"predictions": [{"result": "<json string>"}]}
    resp_json = resp.json()

    # Handle both record-oriented and column-oriented MLflow response formats
    preds = resp_json.get("predictions", resp_json)
    if isinstance(preds, dict):
        # column-oriented: {"result": ["..."]}
        raw = preds["result"][0]
    else:
        # record-oriented: [{"result": "..."}]
        raw = preds[0]["result"]

    data = json.loads(raw)

    # CombinedRealEstateModel wraps per-row errors as {"error": "..."}
    if isinstance(data, list) and data and "error" in data[0]:
        raise ValueError(data[0]["error"])
    if isinstance(data, dict) and "error" in data:
        raise ValueError(data["error"])

    return data



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True})


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
        data = dict(request.json)
        # Derive hold-period bucket features from years_since_last_sale.
        # The model was trained on these binary flags; without them it always
        # predicts near-zero probability.
        yrs = float(data.get("years_since_last_sale", 0) or 0)
        data["hold_0_2"]       = 1 if yrs < 2  else 0
        data["hold_2_5"]       = 1 if 2  <= yrs < 5  else 0
        data["hold_5_10"]      = 1 if 5  <= yrs < 10 else 0
        data["hold_10_plus"]   = 1 if yrs >= 10 else 0
        data["long_hold"]      = 1 if yrs >= 10 else 0
        data["very_long_hold"] = 1 if yrs >= 15 else 0
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
