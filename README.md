# AI Real Estate Intelligence Platform

A Databricks notebook that trains and serves three ML models on US residential real estate data:

- **AVM** — Automated Valuation Model (XGBoost + LightGBM stacked ensemble)
- **Price Forecast** — Prophet-based price trend forecaster by state and property type
- **Lead Scoring** — LightGBM motivated-seller propensity model

All three models are served through a single Databricks endpoint (`reai-combined`) via a routing pyfunc model.

## Files

| File | Description |
|---|---|
| `real_estate_ai.py` | Single-cell Databricks notebook — run this end to end |
| `realtor-data.csv` | Source dataset (2.2M US property listings) |
| `requirements.txt` | Python dependencies |

## Setup

1. Upload `real_estate_ai.py` and `realtor-data.csv` to the same folder in your Databricks workspace (e.g. `Real Estate AI/`)
2. Open `real_estate_ai.py` as a notebook in Databricks
3. Attach a cluster (Databricks Runtime 14+, single-node is fine)
4. Click **Run All**

The notebook auto-detects the CSV path from the notebook's workspace folder — no manual path configuration needed.

## Pipeline

```
Bronze  →  Silver  →  Gold  →  AVM + Forecast + Lead Scoring  →  Combined Endpoint
```

1. **Bronze** — reads CSV into a Delta table
2. **Silver** — cleans, filters, and enriches listings
3. **Gold** — feature engineering, writes feature tables
4. **AVM** — trains stacked ensemble, scores active listings
5. **Price Forecast** — fits Prophet models per state/property type
6. **Lead Scoring** — trains calibrated LightGBM classifier
7. **Combined model** — packages all three into one MLflow pyfunc, registers as `reai_combined`
8. **Investment Analysis** — cap rate, IRR, NPV, Monte Carlo per active listing

## Serving

After the pipeline completes, set the `Promote models to Production?` widget to `true` and re-run to promote `reai_combined` to Production. Then create the endpoint manually in the Databricks UI:

**Machine Learning → Models → reai_combined → Use model for inference**
- Name: `reai-combined`
- Compute: CPU / Small / Scale to zero

### Example requests

```json
// AVM
{"dataframe_records": [{"model_type": "avm", "payload": "[{\"bed\":3,\"bath\":2,\"house_size\":1800,\"state\":\"California\",\"zip_code\":\"90210\"}]"}]}

// Lead scoring
{"dataframe_records": [{"model_type": "lead_scoring", "payload": "[{\"years_since_last_sale\":8,\"state\":\"Texas\",\"property_type\":\"SFR\"}]"}]}

// Price forecast
{"dataframe_records": [{"model_type": "price_forecast", "payload": "[{\"state\":\"California\",\"property_type\":\"SFR\",\"horizon\":12}]"}]}
```

## Catalog layout

All tables are written to the `workspace` Unity Catalog:

```
workspace.reai_bronze.listings
workspace.reai_silver.sold_listings / active_listings / market_snapshots
workspace.reai_gold.avm_estimates / rent_estimates / lead_scores / investment_scores / price_forecasts
workspace.reai_features.property_features / market_features / lead_features
```
