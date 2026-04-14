import os, json, requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST", "https://dbc-496010ba-2046.cloud.databricks.com")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
ENDPOINT_URL     = f"{DATABRICKS_HOST}/serving-endpoints/reai-combined/invocations"

# Pre-filled default inputs matching the form defaults
_AVM_DEFAULT  = {"bed":3,"bath":2,"house_size":1800,"acre_lot":0.35,
                 "state":"California","zip_code":"90210","property_type":"SFR"}
_LEAD_DEFAULT = {"years_since_last_sale":7.5,"price":450000,"bed":3,
                 "house_size":1800,"state":"California","property_type":"SFR",
                 "hold_0_2":0,"hold_2_5":0,"hold_5_10":1,"hold_10_plus":0,
                 "long_hold":0,"very_long_hold":0}
_FORECAST_DEFAULT = {"state":"California","property_type":"SFR","horizon":12}

# In-memory cache: keyed by (model_type, frozenset of input items)
_cache = {}

def _cache_key(model_type, payload):
    return (model_type, json.dumps(payload, sort_keys=True))

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

    preds = resp.json().get("predictions", resp.json())
    if isinstance(preds, dict):
        raw = preds["result"][0]
    else:
        raw = preds[0]["result"]

    data = json.loads(raw)

    if isinstance(data, list) and data and "error" in data[0]:
        raise ValueError(data[0]["error"])
    if isinstance(data, dict) and "error" in data:
        raise ValueError(data["error"])

    return data


def call_cached(model_type: str, payload: list):
    key = _cache_key(model_type, payload)
    if key in _cache:
        return _cache[key]
    result = call_endpoint(model_type, payload)
    _cache[key] = result
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    # Warms up Databricks and pre-populates cache with default inputs.
    try:
        _cache[_cache_key("avm",          [_AVM_DEFAULT])]      = call_endpoint("avm",          [_AVM_DEFAULT])
        _cache[_cache_key("lead_scoring", [_LEAD_DEFAULT])]     = call_endpoint("lead_scoring", [_LEAD_DEFAULT])
        _cache[_cache_key("price_forecast",[_FORECAST_DEFAULT])] = call_endpoint("price_forecast",[_FORECAST_DEFAULT])
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/avm", methods=["POST"])
def avm():
    try:
        data   = request.json
        result = call_cached("avm", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/lead", methods=["POST"])
def lead():
    try:
        data = dict(request.json)
        yrs = float(data.get("years_since_last_sale", 0) or 0)
        data["hold_0_2"]       = 1 if yrs < 2  else 0
        data["hold_2_5"]       = 1 if 2  <= yrs < 5  else 0
        data["hold_5_10"]      = 1 if 5  <= yrs < 10 else 0
        data["hold_10_plus"]   = 1 if yrs >= 10 else 0
        data["long_hold"]      = 1 if yrs >= 10 else 0
        data["very_long_hold"] = 1 if yrs >= 15 else 0
        result = call_cached("lead_scoring", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/forecast", methods=["POST"])
def forecast():
    try:
        data   = request.json
        result = call_cached("price_forecast", [data])
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
