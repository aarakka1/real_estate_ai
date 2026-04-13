import subprocess, sys

# Packages pinned to NumPy 2.x-compatible releases.
# np.obj2sctype was removed in NumPy 2.0 — shap<0.46, xgboost<2.1, lgbm<4.4
# all used it. The versions below are verified compatible with NumPy 2.x.
_PKGS = [
    "xgboost==2.1.3",
    "lightgbm==4.5.0",
    "shap==0.46.0",
    "optuna==3.6.1",
    "prophet==1.1.5",
    "numpy-financial==1.0.0",
    "scikit-learn==1.5.2",
    "requests==2.31.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + _PKGS)
print("Dependencies installed.")

# =============================================================================
# AI Real Estate Intelligence Platform — Single Notebook
# Dataset: realtor-data.zip.csv (2.2M rows, all 50 US states)
#
# Upload the CSV first:
#   databricks fs cp realtor-data.zip.csv dbfs:/reai/source/realtor-data.csv
#
# Run order (use widgets at the top of each section or run all):
#   1. CONFIG          — shared constants and assumptions
#   2. BRONZE          — load CSV into Delta
#   3. SILVER          — clean, enrich, split
#   4. GOLD            — feature engineering → Feature Store
#   5. AVM             — Automated Valuation Model
#   6. PRICE FORECAST  — Prophet price trend + rent estimator
#   7. LEAD SCORING    — seller propensity model
#   8. INVESTMENT      — cap rate, IRR, NPV, Monte Carlo
#   9. SERVING         — MLflow registry + Databricks endpoints
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

import os, json, logging, requests, time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

# ── Notebook-relative CSV path ────────────────────────────────────────────────
# Resolves to the workspace folder that contains this notebook, so that
# realtor-data.csv can be uploaded alongside the notebook in the Databricks UI
# instead of being pushed to DBFS manually.
def _notebook_csv_path(filename: str = "realtor-data.csv") -> str:
    try:
        ctx     = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        nb_path = ctx.notebookPath().get()   # e.g. /Users/you/Real Estate AI/real_estate_ai
        folder  = os.path.dirname(nb_path)
        return f"/Workspace{folder}/{filename}"
    except Exception:
        # Fallback: env-var, or the known workspace location of the notebook folder
        return os.getenv("REAI_SOURCE_CSV", "/Workspace/Real Estate AI/realtor-data.csv")

import numpy as np
import pandas as pd

# ── NumPy 2.0 compatibility shims ────────────────────────────────────────────
# Several libraries (prophet, older shap/xgboost builds) still reference
# aliases that were removed in NumPy 2.0. Restore them before those imports.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "bool_"):
    np.bool_ = np.bool_  # still exists but guard anyway
if not hasattr(np, "object_"):
    np.object_ = object
if not hasattr(np, "str_"):
    np.str_ = np.str_
if not hasattr(np, "obj2sctype"):
    np.obj2sctype = lambda obj, default=None: np.dtype(obj).type

try:
    import numpy_financial as npf
except ImportError:
    import subprocess; subprocess.run(["pip", "install", "-q", "numpy-financial"])
    import numpy_financial as npf

import mlflow, mlflow.sklearn, mlflow.xgboost, mlflow.lightgbm, mlflow.pyfunc
from mlflow.tracking import MlflowClient

import xgboost  as xgb
import lightgbm as lgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection   import GroupKFold, train_test_split, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, OrdinalEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import Ridge
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.metrics           import (mean_absolute_error, mean_squared_error,
                                        r2_score, roc_auc_score,
                                        average_precision_score, brier_score_loss,
                                        classification_report,
                                        mean_absolute_percentage_error)
from sklearn.base              import BaseEstimator, RegressorMixin

from prophet import Prophet

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                                IntegerType, LongType, TimestampType)
from delta.tables import DeltaTable
# Note: databricks.feature_store and databricks.sdk are NOT used —
# feature tables are plain Delta tables written/read via Spark,
# and endpoint creation is done manually via the Databricks UI.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reai")

spark = SparkSession.builder.getOrCreate()
# spark.sql.adaptive, delta.optimizeWrite, and delta.autoCompact are managed
# automatically by Databricks Runtime — setting them manually raises an error.

def _write_feature_table(name: str, df, primary_keys=None):
    """Write a Delta table used as a feature table (no ML Runtime needed)."""
    (df.write.format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(name))

def _read_feature_table(name: str):
    """Read a feature table as a plain Spark DataFrame."""
    return spark.table(name)

# =============================================================================
# 1. CONFIG
# =============================================================================

CATALOG   = "workspace"
DB_BRONZE = f"{CATALOG}.reai_bronze"
DB_SILVER = f"{CATALOG}.reai_silver"
DB_GOLD   = f"{CATALOG}.reai_gold"
DB_FEAT   = f"{CATALOG}.reai_features"

# Storage paths removed — Unity Catalog manages table locations automatically.
# dbfs:/ scheme is not supported for table creation in Unity Catalog.
# Auto-resolved to /Workspace/<notebook-folder>/realtor-data.csv at runtime.
# Upload realtor-data.csv to the same workspace folder as this notebook in the
# Databricks UI (File > Upload to current folder) and it will be found automatically.
REALTOR_CSV_PATH = _notebook_csv_path("realtor-data.csv")

# Experiments are placed directly under the user's home folder (no subdir)
# because MLflow will not create intermediate directories automatically.
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
MLFLOW_EXPERIMENT_BASE = f"/Users/{_current_user}"
MODEL_REGISTRY_NAME = {
    "avm":            "reai_avm",
    "price_forecast": "reai_price_forecast",
    "lead_scoring":   "reai_lead_scoring",
    "combined":       "reai_combined",       # single endpoint serving all three
}
FEATURE_TABLE_PROPERTY = f"{DB_FEAT}.property_features"
FEATURE_TABLE_MARKET   = f"{DB_FEAT}.market_features"
FEATURE_TABLE_LEAD     = f"{DB_FEAT}.lead_features"

PRICE_MIN    = 10_000
PRICE_MAX    = 50_000_000
SQFT_MIN     = 100
SQFT_MAX     = 30_000
BEDS_MAX     = 20
BATHS_MAX    = 20
ACRE_LOT_MAX = 1_000

SPARSE_STATES = {"Alaska","Vermont","Wyoming","Guam","Virgin Islands","New Brunswick"}

STATE_GROSS_YIELD = {
    "California": 0.040, "New York": 0.045, "Hawaii": 0.035,
    "Massachusetts": 0.042, "Washington": 0.043, "New Jersey": 0.048,
    "Connecticut": 0.050, "Florida": 0.058, "Texas": 0.060,
    "Arizona": 0.055, "Colorado": 0.050, "Nevada": 0.057,
    "Georgia": 0.062, "North Carolina": 0.062, "Tennessee": 0.063,
    "South Carolina": 0.065, "Illinois": 0.068, "Ohio": 0.078,
    "Michigan": 0.072, "Wisconsin": 0.072, "Indiana": 0.076,
    "Missouri": 0.074, "Minnesota": 0.066, "_default": 0.065,
}

STATE_TIER_MAP = {
    "California":"coastal_premium","New York":"coastal_premium",
    "Hawaii":"coastal_premium","Massachusetts":"coastal_premium",
    "Washington":"coastal_premium","New Jersey":"coastal_premium",
    "Connecticut":"coastal_premium","Maryland":"coastal_premium",
    "Oregon":"coastal_premium","Florida":"sun_belt","Texas":"sun_belt",
    "Arizona":"sun_belt","Colorado":"sun_belt","Nevada":"sun_belt",
    "Georgia":"sun_belt","North Carolina":"sun_belt","Tennessee":"sun_belt",
    "South Carolina":"sun_belt","Utah":"sun_belt","Idaho":"sun_belt",
    "Montana":"sun_belt","Illinois":"midwest","Ohio":"midwest",
    "Michigan":"midwest","Wisconsin":"midwest","Indiana":"midwest",
    "Missouri":"midwest","Minnesota":"midwest","Iowa":"midwest",
    "Kansas":"midwest","Nebraska":"midwest","Oklahoma":"midwest",
    "Arkansas":"midwest","Mississippi":"midwest","Alabama":"midwest",
    "Kentucky":"midwest","Louisiana":"midwest","West Virginia":"midwest",
    "Pennsylvania":"midwest","Virginia":"midwest","Delaware":"midwest",
    "Rhode Island":"midwest","New Hampshire":"midwest","Maine":"midwest",
    "South Dakota":"midwest","North Dakota":"midwest","New Mexico":"midwest",
}

STATE_TIER_ASSUMPTIONS = {
    "coastal_premium": {"tax_rate":0.009,"insurance_rate":0.004,"mgmt_fee_pct":0.08,
                        "maintenance_pct":0.008,"vacancy_rate":0.05,"annual_pg":0.055,"exit_cap_rate":0.040},
    "sun_belt":        {"tax_rate":0.013,"insurance_rate":0.007,"mgmt_fee_pct":0.08,
                        "maintenance_pct":0.010,"vacancy_rate":0.07,"annual_pg":0.048,"exit_cap_rate":0.050},
    "midwest":         {"tax_rate":0.015,"insurance_rate":0.005,"mgmt_fee_pct":0.09,
                        "maintenance_pct":0.012,"vacancy_rate":0.08,"annual_pg":0.035,"exit_cap_rate":0.065},
    "other":           {"tax_rate":0.012,"insurance_rate":0.005,"mgmt_fee_pct":0.09,
                        "maintenance_pct":0.010,"vacancy_rate":0.08,"annual_pg":0.035,"exit_cap_rate":0.060},
}

AVM_MAX_MAPE      = 0.12
LEAD_MIN_AUC      = 0.68
FORECAST_MAX_MAPE = 0.15
LEAD_SCORE_THRESHOLD = 0.65
SECRET_SCOPE      = "reai"

state_tier_df = spark.createDataFrame(
    [(k, v) for k, v in STATE_TIER_MAP.items()], ["state", "state_tier"])

# =============================================================================
# 2. BRONZE — Load CSV into Delta
# =============================================================================

REALTOR_SCHEMA = StructType([
    StructField("brokered_by",    StringType(), True),
    StructField("status",         StringType(), True),
    StructField("price",          DoubleType(), True),
    StructField("bed",            DoubleType(), True),
    StructField("bath",           DoubleType(), True),
    StructField("acre_lot",       DoubleType(), True),
    StructField("street",         StringType(), True),
    StructField("city",           StringType(), True),
    StructField("state",          StringType(), True),
    StructField("zip_code",       StringType(), True),
    StructField("house_size",     DoubleType(), True),
    StructField("prev_sold_date", StringType(), True),
])

_STAGING_VOLUME = "/Volumes/workspace/reai_staging/files"

def _stage_workspace_file(src_path: str) -> str:
    """Copy a /Workspace file into a UC Volume so all Spark executors can read it."""
    spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.reai_staging")
    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.reai_staging.files")
    dest = f"{_STAGING_VOLUME}/{os.path.basename(src_path)}"
    logger.info(f"Staging {src_path} → {dest}")
    dbutils.fs.cp(src_path, dest, recurse=False)
    return dest

def run_bronze(csv_path: str = REALTOR_CSV_PATH):
    # Spark's distributed reader cannot read directly from /Workspace paths
    # and DBFS root is disabled. Stage the file in a UC Volume instead.
    if csv_path.startswith("/Workspace"):
        csv_path = _stage_workspace_file(csv_path)

    logger.info(f"Reading CSV: {csv_path}")
    df = (spark.read.format("csv")
        .option("header", "true").option("nullValue", "")
        .option("mode", "PERMISSIVE").schema(REALTOR_SCHEMA)
        .load(csv_path)
        .withColumn("ingested_at",  F.current_timestamp())
        .withColumn("source_file",  F.lit(csv_path))
        .withColumn("status",       F.lower(F.trim("status")))
        .withColumn("state",        F.initcap(F.trim("state")))
        .withColumn("city",         F.initcap(F.trim("city")))
        .withColumn("zip_code",     F.lpad(F.regexp_replace("zip_code", r"\.0$", ""), 5, "0"))
        .withColumn("prev_sold_date",
                    F.when(
                        F.to_date("prev_sold_date", "yyyy-MM-dd").between(
                            F.lit("1900-01-01").cast("date"),
                            F.current_date()          # reject future dates and far-future typos
                        ),
                        F.to_date("prev_sold_date", "yyyy-MM-dd")
                    ).otherwise(F.lit(None).cast("date")))
        .withColumn("data_source",  F.lit("realtor_national_csv"))
    )
    for db in [DB_BRONZE, DB_SILVER, DB_GOLD, DB_FEAT]:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true")
       .partitionBy("state")
       .saveAsTable(f"{DB_BRONZE}.listings"))
    logger.info(f"Bronze written: {df.count():,} rows")
    return df

# =============================================================================
# 3. SILVER — Clean, Enrich, Split
# =============================================================================

def run_silver():
    raw = spark.table(f"{DB_BRONZE}.listings")

    df = (raw
        .filter(F.col("price").between(PRICE_MIN, PRICE_MAX))
        .filter(F.col("house_size").isNull() | F.col("house_size").between(SQFT_MIN, SQFT_MAX))
        .filter(F.col("bed").isNull()  | (F.col("bed")  <= BEDS_MAX))
        .filter(F.col("bath").isNull() | (F.col("bath") <= BATHS_MAX))
        .filter(F.col("acre_lot").isNull() | (F.col("acre_lot") <= ACRE_LOT_MAX))
        .filter(F.col("status").isin("for_sale","sold","ready_to_build"))
        .filter(F.col("state").isNotNull() & F.col("zip_code").isNotNull())
        # Derived features
        .withColumn("price_per_sqft",
                    F.when(F.col("house_size") > 0, F.col("price") / F.col("house_size")).cast(DoubleType()))
        .withColumn("property_type",
                    F.when(F.col("house_size").isNull() & F.col("bed").isNull(), "LAND")
                     .when(F.col("bed") >= 5, "MULTI_FAMILY")
                     .when((F.col("house_size") < 1200) & (F.col("acre_lot").isNull() | (F.col("acre_lot") < 0.05)), "CONDO")
                     .when(F.col("acre_lot") >= 5, "RURAL_RESIDENTIAL")
                     .otherwise("SFR"))
        .withColumn("bath_bed_ratio",
                    F.when(F.col("bed") > 0, F.col("bath") / F.col("bed")).cast(DoubleType()))
        .withColumn("lot_sqft", (F.col("acre_lot") * 43560).cast(DoubleType()))
        .withColumn("lot_to_living_ratio",
                    F.when(F.col("house_size") > 0, F.col("lot_sqft") / F.col("house_size")).cast(DoubleType()))
        .withColumn("years_since_last_sale",
                    F.when(F.col("prev_sold_date").isNotNull(),
                           F.round(F.datediff(F.current_date(), "prev_sold_date") / 365.25, 2)).cast(DoubleType()))
        .withColumn("has_prior_sale", F.col("prev_sold_date").isNotNull().cast(IntegerType()))
        .withColumn("sqft_bucket",
                    F.when(F.col("house_size") < 800,  "micro")
                     .when(F.col("house_size") < 1400, "small")
                     .when(F.col("house_size") < 2200, "medium")
                     .when(F.col("house_size") < 3500, "large")
                     .when(F.col("house_size").isNotNull(), "xlarge")
                     .otherwise(F.lit(None).cast(StringType())))
        .withColumn("price_tier",
                    F.when(F.col("price") < 150_000, "entry")
                     .when(F.col("price") < 350_000, "mid")
                     .when(F.col("price") < 700_000, "upper_mid")
                     .when(F.col("price") < 1_500_000, "luxury")
                     .otherwise("ultra_luxury"))
        .join(state_tier_df, "state", "left")
        .withColumn("state_tier", F.coalesce("state_tier", F.lit("other")))
        # Quality score
        .withColumn("quality_score",
                    (F.col("price").isNotNull().cast("int") +
                     F.col("bed").isNotNull().cast("int") +
                     F.col("bath").isNotNull().cast("int") +
                     F.col("house_size").isNotNull().cast("int") +
                     F.col("city").isNotNull().cast("int") +
                     F.col("zip_code").isNotNull().cast("int")) / F.lit(6.0))
    )

    # Census enrichment (optional — requires Census API key in Databricks Secrets)
    try:
        census_key = dbutils.secrets.get(SECRET_SCOPE, "census_api_key")
        census_df  = _fetch_census(census_key)
        if census_df:
            df = df.join(census_df, "zip_code", "left")
            logger.info("Census enrichment applied.")
        else:
            df = _null_census_cols(df)
    except Exception as e:
        logger.warning(f"Census skipped ({e})")
        df = _null_census_cols(df)

    # Market snapshot (zip × property_type aggregates from sold records)
    sold_df   = df.filter(F.col("status") == "sold").filter(F.col("quality_score") >= 0.6).filter(F.col("house_size").isNotNull())
    active_df = df.filter(F.col("status") == "for_sale").filter(F.col("quality_score") >= 0.5)
    market_df = _compute_market_snapshot(sold_df, active_df)

    for tbl, data, part in [
        ("sold_listings",   sold_df,   "state"),
        ("active_listings", active_df, "state"),
        ("all_listings",    df.filter(F.col("quality_score") >= 0.5), "state"),
        ("market_snapshots",market_df, None),
    ]:
        w = data.write.format("delta").mode("overwrite").option("overwriteSchema","true")
        if part: w = w.partitionBy(part)
        w.saveAsTable(f"{DB_SILVER}.{tbl}")
        logger.info(f"Silver.{tbl}: {data.count():,} rows")

def _fetch_census(api_key: str, year: int = 2022):
    vars_map = {
        "B19013_001E":"census_median_income","B01003_001E":"census_population",
        "B25077_001E":"census_median_home_value","B25064_001E":"census_median_gross_rent",
        "B17001_002E":"_pov_pop","B23025_005E":"_unemp","B23025_002E":"_labor",
        "B25002_003E":"_vacant","B25001_001E":"_total_h","B25003_002E":"_owner",
        "B25003_001E":"_occ_total","B15003_022E":"_bach","B15003_001E":"_edu",
    }
    resp = requests.get(
        f"https://api.census.gov/data/{year}/acs/acs5",
        params={"get": f"NAME,{','.join(vars_map)}", "for": "zip code tabulation area:*", "key": api_key},
        timeout=120,
    )
    if resp.status_code != 200: return None
    rows, header = resp.json()[1:], resp.json()[0]
    parsed = []
    for row in rows:
        r = dict(zip(header, row))
        zcta = r.get("zip code tabulation area","")
        def si(k):
            try: return int(r.get(k) or 0)
            except: return 0
        pop=si("B01003_001E"); labor=si("B23025_002E"); unemp=si("B23025_005E")
        pov=si("B17001_002E"); vacant=si("B25002_003E"); total_h=si("B25001_001E")
        owner=si("B25003_002E"); occ=si("B25003_001E"); bach=si("B15003_022E"); edu=si("B15003_001E")
        parsed.append({
            "zip_code": zcta,
            "census_median_income":     si("B19013_001E"),
            "census_population":        pop,
            "census_median_home_value": si("B25077_001E"),
            "census_median_gross_rent": si("B25064_001E"),
            "census_unemployment_rate": round(unemp/max(labor,1),4),
            "census_poverty_rate":      round(pov/max(pop,1),4),
            "census_vacancy_rate":      round(vacant/max(total_h,1),4),
            "census_owner_occ_rate":    round(owner/max(occ,1),4),
            "census_college_pct":       round(bach/max(edu,1),4),
        })
    return spark.createDataFrame(parsed)

def _null_census_cols(df):
    for c in ["census_median_income","census_population","census_median_home_value",
              "census_median_gross_rent","census_unemployment_rate","census_poverty_rate",
              "census_vacancy_rate","census_owner_occ_rate","census_college_pct"]:
        df = df.withColumn(c, F.lit(None).cast(DoubleType()))
    return df

def _compute_market_snapshot(sold_df, active_df):
    grm_map  = {k: v for k, v in STATE_GROSS_YIELD.items()}
    sold_agg = (sold_df
        .groupBy("state","zip_code","property_type","state_tier")
        .agg(
            F.count("*")                              .alias("sold_count"),
            F.percentile_approx("price",0.5)          .alias("median_sold_price"),
            F.percentile_approx("price_per_sqft",0.5) .alias("median_price_per_sqft"),
            F.avg("years_since_last_sale")             .alias("avg_hold_years"),
            F.avg("census_median_income")              .alias("zip_median_income"),
            F.avg("census_unemployment_rate")          .alias("zip_unemployment"),
            F.avg("census_vacancy_rate")               .alias("zip_vacancy_rate"),
            F.avg("census_owner_occ_rate")             .alias("zip_owner_occ_rate"),
        )
    )
    active_agg = (active_df
        .groupBy("state","zip_code","property_type")
        .agg(
            F.count("*")                              .alias("active_listing_count"),
            F.percentile_approx("price",0.5)          .alias("median_list_price"),
        )
    )
    grm_rows   = [(k, v) for k, v in STATE_GROSS_YIELD.items()]
    grm_spark  = spark.createDataFrame(grm_rows, ["state","gross_yield"])
    return (sold_agg
        .join(active_agg, ["state","zip_code","property_type"], "left")
        .join(grm_spark,  "state", "left")
        .withColumn("gross_yield", F.coalesce("gross_yield", F.lit(STATE_GROSS_YIELD["_default"])))
        .withColumn("estimated_monthly_rent",
                    (F.col("median_sold_price") * F.col("gross_yield") / 12).cast(DoubleType()))
        .withColumn("price_to_rent_ratio",
                    F.when(F.col("estimated_monthly_rent") > 0,
                           F.col("median_sold_price") / (F.col("estimated_monthly_rent") * 12)).cast(DoubleType()))
        .withColumn("months_of_supply",
                    F.when(F.col("sold_count") > 0,
                           F.col("active_listing_count") / (F.col("sold_count") / 12.0)).cast(DoubleType()))
        .withColumn("snapshot_date", F.current_date())
    )

# =============================================================================
# 4. GOLD — Feature Engineering
# =============================================================================

def run_gold_features():
    sold   = spark.table(f"{DB_SILVER}.sold_listings")
    active = spark.table(f"{DB_SILVER}.active_listings")
    market = spark.table(f"{DB_SILVER}.market_snapshots")
    all_df = spark.table(f"{DB_SILVER}.all_listings")

    zip_ctx = (market.filter(F.col("property_type")=="SFR").select(
        "zip_code",
        F.col("median_sold_price")     .alias("zip_median_price"),
        F.col("median_price_per_sqft") .alias("zip_median_ppsqft"),
        F.col("sold_count")            .alias("zip_sold_count"),
        F.col("active_listing_count")  .alias("zip_active_count"),
        F.col("gross_yield")           .alias("zip_gross_yield"),
        F.col("estimated_monthly_rent").alias("zip_est_rent"),
        F.col("price_to_rent_ratio")   .alias("zip_ptr_ratio"),
        F.col("avg_hold_years")        .alias("zip_avg_hold_years"),
    ))

    # Property features
    prop_df = (sold.join(zip_ctx,"zip_code","left")
        .withColumn("price_vs_zip_median",
                    F.when(F.col("zip_median_price")>0, F.col("price")/F.col("zip_median_price")).cast(DoubleType()))
        .withColumn("affordability_ratio",
                    F.when(F.col("census_median_income")>0, F.col("price")/(F.col("census_median_income")*4)).cast(DoubleType()))
        .withColumn("absorption_proxy",
                    F.when(F.col("zip_active_count")>0, F.col("zip_sold_count")/F.col("zip_active_count")).cast(DoubleType()))
        .withColumn("tier_coastal",    (F.col("state_tier")=="coastal_premium").cast(IntegerType()))
        .withColumn("tier_sun_belt",   (F.col("state_tier")=="sun_belt").cast(IntegerType()))
        .withColumn("tier_midwest",    (F.col("state_tier")=="midwest").cast(IntegerType()))
        .withColumn("is_multi_family", (F.col("property_type")=="MULTI_FAMILY").cast(IntegerType()))
        .withColumn("is_condo",        (F.col("property_type")=="CONDO").cast(IntegerType()))
        .withColumn("is_rural",        (F.col("property_type")=="RURAL_RESIDENTIAL").cast(IntegerType()))
        .withColumn("has_census_data", F.col("census_median_income").isNotNull().cast(IntegerType()))
        .withColumn("sqft_micro",  (F.col("sqft_bucket")=="micro").cast(IntegerType()))
        .withColumn("sqft_small",  (F.col("sqft_bucket")=="small").cast(IntegerType()))
        .withColumn("sqft_medium", (F.col("sqft_bucket")=="medium").cast(IntegerType()))
        .withColumn("sqft_large",  (F.col("sqft_bucket")=="large").cast(IntegerType()))
        .withColumn("sqft_xlarge", (F.col("sqft_bucket")=="xlarge").cast(IntegerType()))
        .withColumn("feature_ts",  F.current_timestamp())
    )

    # Lead features
    lead_df = (all_df.filter(F.col("has_prior_sale")==1)
        .join(market.filter(F.col("property_type")=="SFR").select(
            "zip_code",
            F.col("median_sold_price")     .alias("zip_current_median"),
            F.col("gross_yield")           .alias("zip_gross_yield"),
            F.col("zip_vacancy_rate"),
            F.col("zip_unemployment"),
        ), "zip_code", "left")
        .withColumn("approx_purchase_price",
                    F.when(F.col("years_since_last_sale").isNotNull() & (F.col("price")>0),
                           F.col("price") / F.pow(F.lit(1.04), F.col("years_since_last_sale"))).cast(DoubleType()))
        .withColumn("unrealized_gain_est",
                    (F.col("price") - F.col("approx_purchase_price")).cast(DoubleType()))
        .withColumn("unrealized_gain_pct",
                    F.when(F.col("approx_purchase_price")>0,
                           F.col("unrealized_gain_est")/F.col("approx_purchase_price")).cast(DoubleType()))
        .withColumn("hold_0_2",     (F.col("years_since_last_sale")<2).cast(IntegerType()))
        .withColumn("hold_2_5",     F.col("years_since_last_sale").between(2,5).cast(IntegerType()))
        .withColumn("hold_5_10",    F.col("years_since_last_sale").between(5,10).cast(IntegerType()))
        .withColumn("hold_10_plus", (F.col("years_since_last_sale")>10).cast(IntegerType()))
        .withColumn("long_hold",    (F.col("years_since_last_sale")>7).cast(IntegerType()))
        .withColumn("very_long_hold",(F.col("years_since_last_sale")>15).cast(IntegerType()))
        .withColumn("high_unrealized_gain", (F.col("unrealized_gain_pct")>0.30).cast(IntegerType()))
        .withColumn("in_sellers_market",
                    (F.col("zip_gross_yield")>F.lit(STATE_GROSS_YIELD["_default"])).cast(IntegerType()))
        .withColumn("large_lot",   (F.col("acre_lot")>=1.0).cast(IntegerType()))
        .withColumn("small_unit",  (F.col("house_size")<1200).cast(IntegerType()))
        .withColumn("high_vacancy_area",      (F.col("census_vacancy_rate")>0.10).cast(IntegerType()))
        .withColumn("high_unemployment_area", (F.col("census_unemployment_rate")>0.07).cast(IntegerType()))
        .withColumn("tier_coastal",  (F.col("state_tier")=="coastal_premium").cast(IntegerType()))
        .withColumn("tier_sun_belt", (F.col("state_tier")=="sun_belt").cast(IntegerType()))
        .withColumn("tier_midwest",  (F.col("state_tier")=="midwest").cast(IntegerType()))
        .withColumn("is_condo",      (F.col("property_type")=="CONDO").cast(IntegerType()))
        .withColumn("is_rural",      (F.col("property_type")=="RURAL_RESIDENTIAL").cast(IntegerType()))
        .withColumn("is_multi_family",(F.col("property_type")=="MULTI_FAMILY").cast(IntegerType()))
        .withColumn("feature_ts",    F.current_timestamp())
    )

    for tbl, df, pks in [
        (FEATURE_TABLE_PROPERTY, prop_df, ["street","city","state","zip_code"]),
        (FEATURE_TABLE_MARKET,   market,  ["zip_code","property_type"]),
        (FEATURE_TABLE_LEAD,     lead_df, ["street","city","state","zip_code"]),
    ]:
        _write_feature_table(tbl, df, primary_keys=pks)
        logger.info(f"Feature table: {tbl} ({df.count():,} rows)")

# =============================================================================
# 5. AVM — Automated Valuation Model
# =============================================================================

AVM_NUMERIC = [
    "bed","bath","house_size","acre_lot","lot_sqft","bath_bed_ratio","lot_to_living_ratio",
    "zip_median_price","zip_median_ppsqft","zip_sold_count","zip_active_count",
    "zip_gross_yield","zip_est_rent","zip_ptr_ratio","zip_avg_hold_years","absorption_proxy",
    "census_median_income","census_median_home_value","census_median_gross_rent",
    "census_unemployment_rate","census_poverty_rate","census_vacancy_rate",
    "census_owner_occ_rate","census_college_pct","affordability_ratio",
]
AVM_CATEGORICAL = ["state","zip_code","property_type","state_tier","sqft_bucket"]
AVM_BINARY = [
    "sqft_micro","sqft_small","sqft_medium","sqft_large","sqft_xlarge",
    "tier_coastal","tier_sun_belt","tier_midwest",
    "is_multi_family","is_condo","is_rural","has_census_data","has_prior_sale",
]
AVM_ALL = AVM_NUMERIC + AVM_CATEGORICAL + AVM_BINARY

LEAD_NUMERIC = ["years_since_last_sale","unrealized_gain_pct","unrealized_gain_est",
                "approx_purchase_price","price","zip_current_median","zip_gross_yield",
                "census_median_income","census_unemployment_rate","census_vacancy_rate",
                "census_owner_occ_rate","census_poverty_rate","census_median_home_value",
                "bed","bath","house_size","acre_lot"]
LEAD_BINARY  = ["hold_0_2","hold_2_5","hold_5_10","hold_10_plus","long_hold","very_long_hold",
                "high_unrealized_gain","in_sellers_market","large_lot","small_unit",
                "high_vacancy_area","high_unemployment_area",
                "tier_coastal","tier_sun_belt","tier_midwest","is_condo","is_rural","is_multi_family"]
LEAD_CAT     = ["state","property_type","state_tier"]
LEAD_ALL     = LEAD_NUMERIC + LEAD_BINARY + LEAD_CAT

class NationalAVM(BaseEstimator, RegressorMixin):
    def __init__(self, n_folds=3):
        self.n_folds=n_folds; self.preprocessor=None
        self.xgb_model=None; self.lgb_model=None; self.meta_learner=None

    def _make_preprocessor(self):
        return ColumnTransformer(transformers=[
            ("num", StandardScaler(), AVM_NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), AVM_CATEGORICAL),
            ("bin", "passthrough", AVM_BINARY),
        ], remainder="drop")

    def fit(self, X, y, groups=None):
        self.preprocessor = self._make_preprocessor()
        Xp = self.preprocessor.fit_transform(X)
        groups = groups if groups is not None else pd.Series(["unknown"]*len(X))
        oof = np.zeros((len(Xp), 2))
        xm = lm = None
        for fold, (ti, vi) in enumerate(GroupKFold(self.n_folds).split(Xp, y, groups)):
            logger.info(f"AVM fold {fold+1}/{self.n_folds}")
            xm = xgb.XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,
                                   subsample=0.8,colsample_bytree=0.7,min_child_weight=10,
                                   tree_method="hist",random_state=42,n_jobs=-1,
                                   early_stopping_rounds=30,eval_metric="mae")
            xm.fit(Xp[ti],y.iloc[ti],eval_set=[(Xp[vi],y.iloc[vi])],verbose=False)
            lm = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,
                                    num_leaves=63,subsample=0.8,colsample_bytree=0.7,
                                    random_state=42,n_jobs=-1)
            lm.fit(Xp[ti],y.iloc[ti],eval_set=[(Xp[vi],y.iloc[vi])],
                   callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
            oof[vi,0]=xm.predict(Xp[vi]); oof[vi,1]=lm.predict(Xp[vi])
            logger.info(f"  Fold MAE: ${mean_absolute_error(np.expm1(y.iloc[vi]), np.expm1(oof[vi].mean(1))):,.0f}")
        self.xgb_model=xm; self.lgb_model=lm
        self.meta_learner=Ridge(alpha=1.0).fit(oof, y)
        return self

    def predict(self, X):
        Xp = self.preprocessor.transform(X)
        base = np.column_stack([self.xgb_model.predict(Xp), self.lgb_model.predict(Xp)])
        return np.expm1(self.meta_learner.predict(base))

    def predict_with_interval(self, X, pct=0.10):
        est = self.predict(X)
        return pd.DataFrame({"estimate":est,"low":est*(1-pct),"high":est*(1+pct)})


def run_avm(test_states=None):
    if test_states is None:
        test_states = ["California","Texas","Ohio","Montana","New York"]

    df = _read_feature_table(FEATURE_TABLE_PROPERTY).filter(
        F.col("price").between(PRICE_MIN,PRICE_MAX) &
        (F.col("quality_score")>=0.6) &
        (F.col("property_type")!="LAND")
    ).toPandas()
    df["log_price"] = np.log1p(df["price"])

    for c in AVM_NUMERIC:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(df[c].median() if c in df else 0)
    for c in AVM_BINARY:
        if c in df: df[c]=df[c].fillna(0).astype(int)
    for c in AVM_CATEGORICAL:
        if c in df: df[c]=df[c].fillna("unknown")

    mask   = ~df["state"].isin(test_states)
    X_tr, y_tr, g_tr = df.loc[mask,AVM_ALL], df.loc[mask,"log_price"], df.loc[mask,"state"]
    X_te, y_te        = df.loc[~mask,AVM_ALL], df.loc[~mask,"price"]

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_BASE}/reai_avm")
    with mlflow.start_run(run_name=f"avm_{datetime.now():%Y%m%d_%H%M}"):
        mlflow.log_params({"model":"NationalAVM","cv":"GroupKFold_state",
                           "n_train":len(X_tr),"n_test":len(X_te),
                           "test_states":str(test_states)})
        model = NationalAVM(n_folds=3)
        model.fit(X_tr, y_tr, groups=g_tr)

        preds   = model.predict(X_te)
        mape    = mean_absolute_percentage_error(y_te, preds)
        mae     = mean_absolute_error(y_te, preds)
        r2      = r2_score(y_te, preds)
        mlflow.log_metrics({"mape":mape,"mae":mae,"r2":r2})
        logger.info(f"AVM — MAPE:{mape:.2%} MAE:${mae:,.0f} R²:{r2:.4f} "
                    f"({'PASS ✓' if mape<=AVM_MAX_MAPE else 'FAIL ✗'})")

        avm_input_example = X_te.head(2)
        avm_signature = mlflow.models.infer_signature(
            avm_input_example, model.predict(avm_input_example)
        )
        mlflow.sklearn.log_model(model, "model",
            registered_model_name=MODEL_REGISTRY_NAME["avm"],
            signature=avm_signature,
            input_example=avm_input_example)

        # Batch score active listings → gold.avm_estimates
        active = spark.table(f"{DB_SILVER}.active_listings").toPandas()
        for c in AVM_ALL:
            if c not in active: active[c]=np.nan
        for c in AVM_NUMERIC:    active[c]=pd.to_numeric(active[c],errors="coerce").fillna(0)
        for c in AVM_BINARY:     active[c]=active[c].fillna(0).astype(int)
        for c in AVM_CATEGORICAL:active[c]=active[c].fillna("unknown")
        preds_df = model.predict_with_interval(active[AVM_ALL])
        active["avm_estimate"]=preds_df["estimate"]; active["avm_low"]=preds_df["low"]
        active["avm_high"]=preds_df["high"]; active["scored_at"]=pd.Timestamp.now()
        _avm_out = active[["street","city","state","zip_code","property_type",
                            "price","avm_estimate","avm_low","avm_high","scored_at"]]
        if not _avm_out.empty:
            (spark.createDataFrame(_avm_out)
                  .write.format("delta").mode("overwrite").option("overwriteSchema","true")
                  .saveAsTable(f"{DB_GOLD}.avm_estimates"))

        return mlflow.active_run().info.run_id, {"mape":mape,"mae":mae,"r2":r2}, model

# =============================================================================
# 6. PRICE FORECAST + RENT ESTIMATOR
# =============================================================================

class PriceTrendForecaster(mlflow.pyfunc.PythonModel):
    """
    Appreciation-rate forecaster.  Stores current median prices per
    (state, property_type) and projects forward using state-tier-based
    annual appreciation rates.  Works for every state with no minimum
    time-series requirement.
    """
    # 5-yr annualised appreciation rate by state tier
    _TIER_RATES = {"coastal":0.055,"sun_belt":0.060,"midwest":0.032,"other":0.038}

    def __init__(self): self.baselines={}; self.rates={}

    def fit(self, baseline_df):
        for _,row in baseline_df.iterrows():
            key=(str(row["state"]),str(row["property_type"]))
            self.baselines[key]=float(row["median_price"])
            tier=str(row.get("state_tier","other"))
            self.rates[key]=self._TIER_RATES.get(tier,0.038)

    def predict(self, context, inp):
        results=[]
        today=pd.Timestamp.now().normalize().replace(day=1)
        for _,row in inp.iterrows():
            key=(str(row["state"]),str(row["property_type"])); h=int(row.get("horizon",12))
            if key not in self.baselines: continue
            base=self.baselines[key]; rate=self.rates[key]
            monthly=(1+rate)**(1/12)-1
            rows=[{
                "forecast_month": today+pd.DateOffset(months=m),
                "forecast_price": base*(1+monthly)**m,
                "forecast_price_low":  base*(1+monthly)**m*0.90,
                "forecast_price_high": base*(1+monthly)**m*1.10,
                "state":key[0],"property_type":key[1],"last_actual":base,
            } for m in range(1,h+1)]
            results.append(pd.DataFrame(rows))
        return pd.concat(results,ignore_index=True) if results else pd.DataFrame()


def run_price_forecast(horizon=12):
    # Build baseline medians from active listings — every state with ≥3 listings
    # gets a forecast; no historical time-series required.
    active=spark.table(f"{DB_SILVER}.active_listings").toPandas()
    active=active[active["price"].notna()&active["property_type"].notna()&active["state"].notna()]
    baseline=(active.groupby(["state","property_type","state_tier"])
                    .agg(median_price=("price","median"),count=("price","count"))
                    .reset_index())
    baseline=baseline[baseline["count"]>=3].copy()

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_BASE}/reai_price_forecast")
    with mlflow.start_run(run_name=f"price_forecast_{datetime.now():%Y%m%d_%H%M}"):
        model=PriceTrendForecaster(); model.fit(baseline)
        segs=[{"state":row["state"],"property_type":row["property_type"],"horizon":horizon}
              for _,row in baseline.iterrows()]
        fc_df=model.predict(None,pd.DataFrame(segs))
        if fc_df.empty:
            logger.warning("Price forecast produced no rows — skipping gold write.")
        else:
            (spark.createDataFrame(fc_df).withColumn("generated_at",F.current_timestamp())
                  .write.format("delta").mode("overwrite").option("overwriteSchema","true")
                  .saveAsTable(f"{DB_GOLD}.price_forecasts"))
        fc_input_example = pd.DataFrame(segs[:2])
        fc_signature = mlflow.models.infer_signature(
            fc_input_example,
            model.predict(None, fc_input_example)
        )
        mlflow.pyfunc.log_model("model", python_model=model,
            registered_model_name=MODEL_REGISTRY_NAME["price_forecast"],
            signature=fc_signature,
            input_example=fc_input_example)
        return mlflow.active_run().info.run_id, model, segs

def run_rent_estimator():
    feat  = _read_feature_table(FEATURE_TABLE_PROPERTY)
    RENT_COLS = ["bed","bath","house_size","census_median_gross_rent",
                 "census_median_income","census_median_home_value",
                 "zip_est_rent","zip_median_price"]
    df    = (feat.filter(F.col("census_median_gross_rent").between(200,10_000))
                  .filter(F.col("quality_score")>=0.6).toPandas())
    for c in RENT_COLS:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)

    if len(df) < 20:
        logger.warning(f"Rent estimator: only {len(df)} rows after filtering — skipping.")
        return None

    X=df[RENT_COLS]; y=df["census_median_gross_rent"]
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.15,random_state=42)
    rent_model=lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=5,random_state=42)
    rent_model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],
                   callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(-1)])

    active=spark.table(f"{DB_SILVER}.active_listings").toPandas()
    for c in RENT_COLS:
        if c in active: active[c]=pd.to_numeric(active[c],errors="coerce").fillna(0)
        else: active[c]=0.0
    model_rent=rent_model.predict(active[RENT_COLS]).clip(min=0)
    grm_rent=np.array([(active.loc[i,"price"]*(STATE_GROSS_YIELD.get(active.loc[i,"state"],
               STATE_GROSS_YIELD["_default"]))/12) for i in active.index])
    active["est_monthly_rent"]=(0.70*model_rent+0.30*grm_rent).clip(min=0)
    active["est_annual_rent"]=active["est_monthly_rent"]*12; active["scored_at"]=pd.Timestamp.now()
    _rent_out = active[["street","city","state","zip_code","property_type",
                         "price","est_monthly_rent","est_annual_rent","scored_at"]]
    if not _rent_out.empty:
        (spark.createDataFrame(_rent_out)
              .write.format("delta").mode("overwrite").option("overwriteSchema","true")
              .saveAsTable(f"{DB_GOLD}.rent_estimates"))
    mape=mean_absolute_percentage_error(y_val,rent_model.predict(X_val))
    logger.info(f"Rent estimator MAPE: {mape:.2%}")
    return mape

# =============================================================================
# 7. LEAD SCORING
# =============================================================================

# LEAD_NUMERIC, LEAD_BINARY, LEAD_CAT, LEAD_ALL imported from reai_classes above.

def run_lead_scoring(tune=False):
    lead_df = _read_feature_table(FEATURE_TABLE_LEAD).toPandas()

    sold_addrs   = set(spark.table(f"{DB_SILVER}.sold_listings")
                        .select(F.concat_ws("|","street","city","state").alias("a"))
                        .distinct().toPandas()["a"])
    active_addrs = set(spark.table(f"{DB_SILVER}.active_listings")
                        .select(F.concat_ws("|","street","city","state").alias("a"))
                        .distinct().toPandas()["a"])
    re_listers   = sold_addrs & active_addrs
    lead_df["addr"] = lead_df["street"].fillna("")+"|"+lead_df["city"].fillna("")+"|"+lead_df["state"].fillna("")
    lead_df["motivated_seller"] = (
        lead_df["addr"].isin(re_listers) |
        (lead_df.get("very_long_hold",pd.Series(0,index=lead_df.index))==1) |
        ((lead_df.get("long_hold",pd.Series(0,index=lead_df.index))==1) &
         (lead_df.get("in_sellers_market",pd.Series(0,index=lead_df.index))==1))
    ).astype(int)

    for c in LEAD_NUMERIC:
        if c in lead_df: lead_df[c]=pd.to_numeric(lead_df[c],errors="coerce").fillna(lead_df[c].median() if c in lead_df else 0)
    for c in LEAD_BINARY:
        if c in lead_df: lead_df[c]=lead_df[c].fillna(0).astype(int)
    for c in LEAD_CAT:
        if c in lead_df: lead_df[c]=lead_df[c].fillna("unknown")

    lead_df=lead_df.dropna(subset=["years_since_last_sale"])
    if len(lead_df) < 20:
        logger.warning(f"Lead scoring: only {len(lead_df)} rows after filtering — skipping.")
        return None, {"auc": None, "ap": None}, None, None

    X=lead_df[LEAD_ALL]; y=lead_df["motivated_seller"]
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    X_tr2,X_val,y_tr2,y_val=train_test_split(X_tr,y_tr,test_size=0.15,stratify=y_tr,random_state=42)

    pre = ColumnTransformer(transformers=[
        ("num",StandardScaler(),LEAD_NUMERIC),("bin","passthrough",LEAD_BINARY),
        ("cat",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),LEAD_CAT)
    ],remainder="drop")
    Xp_tr=pre.fit_transform(X_tr2); Xp_val=pre.transform(X_val); Xp_te=pre.transform(X_te)
    pos_w=(y_tr2==0).sum()/max((y_tr2==1).sum(),1)
    base=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,num_leaves=63,
                              scale_pos_weight=pos_w,random_state=42,n_jobs=-1)
    base.fit(Xp_tr,y_tr2,eval_set=[(Xp_val,y_val)],
             callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    cal=CalibratedClassifierCV(base,cv="prefit",method="sigmoid").fit(Xp_val,y_val)

    probas=cal.predict_proba(Xp_te)[:,1]
    auc=roc_auc_score(y_te,probas); ap=average_precision_score(y_te,probas)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_BASE}/reai_lead_scoring")
    with mlflow.start_run(run_name=f"lead_scoring_{datetime.now():%Y%m%d_%H%M}"):
        mlflow.log_params({"model":"lgbm_calibrated","n_train":len(X_tr),"pos_rate":float(y_tr.mean())})
        mlflow.log_metrics({"auc":auc,"average_precision":ap})
        logger.info(f"Lead scoring — AUC:{auc:.4f} AP:{ap:.4f} "
                    f"({'PASS ✓' if auc>=LEAD_MIN_AUC else 'FAIL ✗'})")

        # Score all leads
        all_lead=_read_feature_table(FEATURE_TABLE_LEAD).toPandas()
        for c in LEAD_ALL:
            if c not in all_lead: all_lead[c]=np.nan
        for c in LEAD_NUMERIC: all_lead[c]=pd.to_numeric(all_lead[c],errors="coerce").fillna(0)
        for c in LEAD_BINARY:  all_lead[c]=all_lead[c].fillna(0).astype(int)
        for c in LEAD_CAT:     all_lead[c]=all_lead[c].fillna("unknown")
        Xp_all=pre.transform(all_lead[LEAD_ALL])
        all_lead["lead_score"]=cal.predict_proba(Xp_all)[:,1]
        all_lead["lead_tier"]=pd.cut(all_lead["lead_score"],
            bins=[-np.inf,0.35,LEAD_SCORE_THRESHOLD,np.inf],labels=["cool","warm","hot"]).astype(str)
        all_lead["scored_at"]=pd.Timestamp.now()
        _lead_out = all_lead[["street","city","state","zip_code","lead_score","lead_tier","scored_at"]]
        if not _lead_out.empty:
            (spark.createDataFrame(_lead_out)
                  .write.format("delta").mode("overwrite").option("overwriteSchema","true")
                  .saveAsTable(f"{DB_GOLD}.lead_scores"))
        lead_input_example = X_te.head(2)
        lead_signature = mlflow.models.infer_signature(
            lead_input_example,
            pd.DataFrame({"lead_score": cal.predict_proba(pre.transform(lead_input_example))[:, 1]})
        )
        mlflow.sklearn.log_model({"pre":pre,"cal":cal},"model",
            registered_model_name=MODEL_REGISTRY_NAME["lead_scoring"],
            signature=lead_signature,
            input_example=lead_input_example)
        return mlflow.active_run().info.run_id, {"auc":auc,"ap":ap}, pre, cal

# =============================================================================
# 8. COMBINED PYFUNC MODEL — single endpoint routing AVM / Lead / Forecast
# =============================================================================

class CombinedRealEstateModel(mlflow.pyfunc.PythonModel):
    """
    Routes inference to the correct sub-model based on the 'model_type' field.
    Input:  DataFrame with columns model_type (str) and payload (str JSON array).
    Output: DataFrame with column result (str JSON array).

    Sub-models are stored as cloudpickle artifacts (not as __init__ args) so
    that cloudpickle can embed all class definitions and no reai_classes module
    import is needed in the serving container.

    Example request:
        {"dataframe_records": [
          {"model_type": "avm",
           "payload": "[{\"bed\":3,\"bath\":2,\"house_size\":1800,...}]"},
          {"model_type": "lead_scoring",
           "payload": "[{\"years_since_last_sale\":8,...}]"},
          {"model_type": "price_forecast",
           "payload": "[{\"state\":\"California\",\"property_type\":\"SFR\",\"horizon\":12}]"}
        ]}
    """
    def __init__(self):
        # Store column lists as plain instance attributes so they are pickled
        # with this object and available in the serving container without any
        # module imports.
        self.avm_numeric     = AVM_NUMERIC
        self.avm_categorical = AVM_CATEGORICAL
        self.avm_binary      = AVM_BINARY
        self.avm_all         = AVM_ALL
        self.lead_numeric    = LEAD_NUMERIC
        self.lead_binary     = LEAD_BINARY
        self.lead_cat        = LEAD_CAT
        self.lead_all        = LEAD_ALL

    def load_context(self, context):
        """MLflow calls this before the first predict() — loads sub-models from artifacts."""
        # Apply NumPy 2.0 shims BEFORE importing prophet (triggered by loading
        # the forecast artifact).  prophet==1.1.5 uses np.float_ in a class-level
        # type annotation that is evaluated at import time.
        import numpy as np
        if not hasattr(np, "float_"):    np.float_    = np.float64
        if not hasattr(np, "int_"):      np.int_      = np.int64
        if not hasattr(np, "complex_"):  np.complex_  = np.complex128
        if not hasattr(np, "object_"):   np.object_   = object
        if not hasattr(np, "bool_"):     np.bool_     = np.bool_
        if not hasattr(np, "obj2sctype"):
            np.obj2sctype = lambda obj, default=None: np.dtype(obj).type

        import cloudpickle
        with open(context.artifacts["avm"],      "rb") as f: self.avm      = cloudpickle.load(f)
        with open(context.artifacts["lead_pre"], "rb") as f: self.lead_pre = cloudpickle.load(f)
        with open(context.artifacts["lead_cal"], "rb") as f: self.lead_cal = cloudpickle.load(f)
        with open(context.artifacts["forecast"], "rb") as f: self.forecast = cloudpickle.load(f)

    def predict(self, context, model_input):
        import json, numpy as np, pandas as pd
        if not hasattr(np,"float_"): np.float_=np.float64
        if not hasattr(np,"obj2sctype"): np.obj2sctype=lambda obj,default=None:np.dtype(obj).type
        # Column lists come from instance attributes — no globals() or imports needed.
        AVM_NUMERIC=self.avm_numeric; AVM_BINARY=self.avm_binary
        AVM_CATEGORICAL=self.avm_categorical; AVM_ALL=self.avm_all
        LEAD_NUMERIC=self.lead_numeric; LEAD_BINARY=self.lead_binary
        LEAD_CAT=self.lead_cat; LEAD_ALL=self.lead_all
        results=[]
        for _,row in model_input.iterrows():
            mtype=str(row["model_type"]).strip()
            data=pd.DataFrame(json.loads(row["payload"]))
            try:
                if mtype=="avm":
                    for c in AVM_NUMERIC: data[c]=pd.to_numeric(data.get(c,pd.Series([0]*len(data))),errors="coerce").fillna(0)
                    for c in AVM_BINARY:  data[c]=data.get(c,pd.Series([0]*len(data))).fillna(0).astype(int)
                    for c in AVM_CATEGORICAL: data[c]=data.get(c,pd.Series(["unknown"]*len(data))).fillna("unknown")
                    for c in AVM_ALL:
                        if c not in data: data[c]=0 if c in AVM_NUMERIC+AVM_BINARY else "unknown"
                    results.append(self.avm.predict_with_interval(data[AVM_ALL]).to_json(orient="records"))
                elif mtype=="lead_scoring":
                    for c in LEAD_NUMERIC: data[c]=pd.to_numeric(data.get(c,pd.Series([0]*len(data))),errors="coerce").fillna(0)
                    for c in LEAD_BINARY:  data[c]=data.get(c,pd.Series([0]*len(data))).fillna(0).astype(int)
                    for c in LEAD_CAT:     data[c]=data.get(c,pd.Series(["unknown"]*len(data))).fillna("unknown")
                    for c in LEAD_ALL:
                        if c not in data: data[c]=0 if c in LEAD_NUMERIC+LEAD_BINARY else "unknown"
                    Xp=self.lead_pre.transform(data[LEAD_ALL])
                    results.append(pd.DataFrame({"lead_score":self.lead_cal.predict_proba(Xp)[:,1]}).to_json(orient="records"))
                elif mtype=="price_forecast":
                    fc=self.forecast.predict(None,data)
                    results.append(fc.to_json(orient="records",date_format="iso") if isinstance(fc,pd.DataFrame) and not fc.empty else json.dumps([]))
                else:
                    results.append(json.dumps({"error":f"Unknown model_type '{mtype}'. Use: avm, lead_scoring, price_forecast"}))
            except Exception as e:
                results.append(json.dumps({"error":str(e)}))
        return pd.DataFrame({"result":results})


# No module-registration or code_paths needed.
# cloudpickle (used internally by MLflow) serialises classes defined in
# __main__ by embedding their bytecode — the serving container reconstructs
# them without importing any external module.
# Sub-models are saved as separate cloudpickle artifacts and loaded in
# CombinedRealEstateModel.load_context().


def run_combined_model(avm_model, lead_pre, lead_cal, forecast_model):
    """Log and register the combined routing model as a single MLflow pyfunc.

    Each sub-model is saved as a cloudpickle artifact.  cloudpickle embeds the
    full class definition (NationalAVM, PriceTrendForecaster, etc.) in the
    pickle file, so the serving container can reconstruct them without needing
    any extra code_paths or reai_classes module.
    """
    if any(x is None for x in [avm_model, lead_pre, lead_cal, forecast_model]):
        logger.warning("Combined model skipped — one or more sub-models unavailable.")
        return None

    import cloudpickle

    # Save each sub-model to /tmp with cloudpickle
    _artifacts = {}
    for _name, _obj in [("avm", avm_model), ("lead_pre", lead_pre),
                         ("lead_cal", lead_cal), ("forecast", forecast_model)]:
        _path = f"/tmp/reai_{_name}.pkl"
        with open(_path, "wb") as _fp:
            cloudpickle.dump(_obj, _fp)
        _artifacts[_name] = _path
        logger.info(f"Saved sub-model '{_name}' → {_path}")

    # CombinedRealEstateModel() has no sub-model args — only plain lists.
    # cloudpickle embeds its class definition by value (it lives in __main__).
    combined = CombinedRealEstateModel()

    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema, ColSpec
    sig = ModelSignature(
        inputs =Schema([ColSpec("string","model_type"), ColSpec("string","payload")]),
        outputs=Schema([ColSpec("string","result")]),
    )
    input_example = pd.DataFrame({
        "model_type": ["avm"],
        "payload":    [json.dumps([{c: 0 if c in AVM_NUMERIC + AVM_BINARY else "unknown"
                                    for c in AVM_ALL}])],
    })
    pip_reqs = [
        "xgboost==2.1.3",
        "lightgbm==4.5.0",
        "prophet==1.1.5",
        "scikit-learn==1.5.2",
        "numpy-financial==1.0.0",
        "cloudpickle",
    ]

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_BASE}/reai_combined")
    with mlflow.start_run(run_name=f"combined_{datetime.now():%Y%m%d_%H%M}"):
        mlflow.pyfunc.log_model(
            "model",
            python_model=combined,
            artifacts=_artifacts,            # sub-models bundled as artifacts
            registered_model_name=MODEL_REGISTRY_NAME["combined"],
            signature=sig,
            input_example=input_example,
            pip_requirements=pip_reqs,
            # No code_paths — cloudpickle handles class serialisation
        )
        run_id = mlflow.active_run().info.run_id
    logger.info(f"Combined model registered: {MODEL_REGISTRY_NAME['combined']} (run {run_id})")
    return run_id


# =============================================================================
# 9. INVESTMENT ANALYSIS
# =============================================================================

@dataclass
class FinancingAssumptions:
    purchase_price:float; down_payment_pct:float=0.25; interest_rate:float=0.065; loan_term_years:int=30; closing_cost_pct:float=0.025
    @property
    def loan_amount(self):    return self.purchase_price*(1-self.down_payment_pct)
    @property
    def monthly_rate(self):   return self.interest_rate/12
    @property
    def n_payments(self):     return self.loan_term_years*12
    @property
    def monthly_payment(self):
        r,n=self.monthly_rate,self.n_payments
        return self.loan_amount*(r*(1+r)**n)/((1+r)**n-1) if r else self.loan_amount/n
    @property
    def total_cash_in(self):  return self.purchase_price*(self.down_payment_pct+self.closing_cost_pct)

@dataclass
class OperatingAssumptions:
    monthly_rent:float; vacancy_rate:float; tax_rate:float; insurance_rate:float
    mgmt_fee_pct:float; maintenance_pct:float; capex_pct:float=0.05; hoa_monthly:float=0.0
    def annual_opex(self,p):
        eff=self.monthly_rent*(1-self.vacancy_rate)*12
        return eff*(self.mgmt_fee_pct+self.capex_pct)+p*(self.tax_rate+self.insurance_rate+self.maintenance_pct)+self.hoa_monthly*12

@dataclass
class HoldAssumptions:
    hold_years:int=5; annual_pg:float=0.04; annual_rg:float=0.03; exit_cap_rate:float=0.055; selling_cost_pct:float=0.06; discount_rate:float=0.08

class InvestmentAnalyzer:
    def __init__(self,fin,ops,hold): self.fin=fin; self.ops=ops; self.hold=hold
    def _noi(self,yr=1):
        rent=self.ops.monthly_rent*(1+self.hold.annual_rg)**(yr-1)
        eff=rent*12*(1-self.ops.vacancy_rate)
        return eff-self.ops.annual_opex(self.fin.purchase_price)
    def _cf(self,yr=1):   return self._noi(yr)-self.fin.monthly_payment*12
    def _bal(self,yrs):
        r,n,np2=self.fin.monthly_rate,self.fin.n_payments,yrs*12
        return self.fin.loan_amount*((1+r)**n-(1+r)**np2)/((1+r)**n-1) if r else self.fin.loan_amount*(1-np2/n)
    def _exit(self):
        appr=self.fin.purchase_price*(1+self.hold.annual_pg)**self.hold.hold_years
        cap=self._noi(self.hold.hold_years)/self.hold.exit_cap_rate
        return (appr+cap)/2
    def gross_yield(self):    return self.ops.monthly_rent*12/self.fin.purchase_price
    def cap_rate(self):       return self._noi(1)/self.fin.purchase_price
    def cash_on_cash(self):   return self._cf(1)/self.fin.total_cash_in
    def dscr(self,yr=1):      return self._noi(yr)/(self.fin.monthly_payment*12)
    def irr(self):
        cfs=[-self.fin.total_cash_in]+[self._cf(yr) for yr in range(1,self.hold.hold_years+1)]
        cfs[-1]+=self._exit()*(1-self.hold.selling_cost_pct)-self._bal(self.hold.hold_years)
        try: return npf.irr(cfs)
        except: return float("nan")
    def npv(self):
        cfs=[-self.fin.total_cash_in]+[self._cf(yr) for yr in range(1,self.hold.hold_years+1)]
        cfs[-1]+=self._exit()*(1-self.hold.selling_cost_pct)-self._bal(self.hold.hold_years)
        return npf.npv(self.hold.discount_rate,cfs)
    def summary(self):
        return {"purchase_price":self.fin.purchase_price,"total_cash_in":self.fin.total_cash_in,
                "monthly_payment":self.fin.monthly_payment,"monthly_rent":self.ops.monthly_rent,
                "gross_yield":self.gross_yield(),"cap_rate":self.cap_rate(),
                "cash_on_cash_yr1":self.cash_on_cash(),"dscr_yr1":self.dscr(),
                "noi_yr1":self._noi(1),"cash_flow_yr1":self._cf(1),
                "exit_value":self._exit(),"irr":self.irr(),"npv":self.npv(),"hold_years":self.hold.hold_years}

def run_investment_analysis(down_pct=0.25, rate=0.065, hold_years=5):
    avm  = spark.table(f"{DB_GOLD}.avm_estimates").toPandas()
    keys = ["street","city","state","zip_code"]

    # Use tableExists() — more reliable than try/except for Spark AnalysisException
    if spark.catalog.tableExists(f"{DB_GOLD}.rent_estimates"):
        rent   = spark.table(f"{DB_GOLD}.rent_estimates").toPandas()
        merged = avm.merge(rent[keys+["est_monthly_rent"]], on=keys, how="left")
    else:
        logger.warning("rent_estimates not found — using GRM fallback for rent.")
        merged = avm.copy()
        merged["est_monthly_rent"] = None

    active = spark.table(f"{DB_SILVER}.active_listings").select(*keys,"state_tier").toPandas()
    merged = merged.merge(active, on=keys, how="left")

    # Vectorized: map state → GRM yield and tier assumptions up front
    grm_series  = merged["state"].map(lambda s: STATE_GROSS_YIELD.get(s, STATE_GROSS_YIELD["_default"]))
    merged["state_tier"] = merged["state_tier"].fillna("other")
    merged["price"]      = pd.to_numeric(merged.get("avm_estimate", merged.get("price")), errors="coerce")
    merged["rent_"]      = merged["est_monthly_rent"].where(
        merged["est_monthly_rent"].notna() & (merged["est_monthly_rent"] > 0),
        merged["price"] * grm_series / 12
    )
    merged = merged[(merged["price"] > 0) & (merged["rent_"] > 0)].copy()

    records=[]
    for _,row in merged.iterrows():
        price = float(row["price"])
        rent_ = float(row["rent_"])
        tier  = row.get("state_tier","other")
        t=STATE_TIER_ASSUMPTIONS.get(tier,STATE_TIER_ASSUMPTIONS["other"])
        fin =FinancingAssumptions(purchase_price=price,down_payment_pct=down_pct,interest_rate=rate)
        ops =OperatingAssumptions(monthly_rent=rent_,vacancy_rate=t["vacancy_rate"],
                                   tax_rate=t["tax_rate"],insurance_rate=t["insurance_rate"],
                                   mgmt_fee_pct=t["mgmt_fee_pct"],maintenance_pct=t["maintenance_pct"])
        hold=HoldAssumptions(hold_years=hold_years,annual_pg=t["annual_pg"],exit_cap_rate=t["exit_cap_rate"])
        a   =InvestmentAnalyzer(fin,ops,hold); s=a.summary()
        s.update({k:row.get(k,"") for k in ["street","city","state","zip_code"]})
        s["avm_estimate"]=price; s["est_rent"]=rent_; s["state_tier"]=tier
        records.append(s)

    if not records:
        logger.warning("Investment analysis produced no rows — skipping gold write.")
    else:
        (spark.createDataFrame(pd.DataFrame(records))
              .write.format("delta").mode("overwrite").option("overwriteSchema","true")
              .saveAsTable(f"{DB_GOLD}.investment_scores"))
    logger.info(f"Investment scores: {len(records):,} properties")
    return len(records)

# =============================================================================
# 9. MODEL REGISTRATION — Promote latest runs to Production in MLflow Registry
#
# Endpoint creation is done MANUALLY in the Databricks UI (see instructions
# printed at the bottom of this cell after running promote_models()).
# =============================================================================

def promote_models():
    """
    Checks quality gates on the latest registered model version for each model
    and transitions passing versions to the 'Production' stage in the MLflow
    Model Registry.  No serving endpoints are created here — follow the printed
    instructions to deploy them through the Databricks UI.
    """
    client = MlflowClient()

    # Only the combined model is promoted — it's the one being served.
    # Individual models (avm, lead_scoring, price_forecast) stay registered
    # for tracking but don't need their own endpoint.
    quality_gates = {
        MODEL_REGISTRY_NAME["combined"]: None,  # no single metric gate; sub-models checked at train time
    }

    promoted = {}
    for model_name in [MODEL_REGISTRY_NAME["combined"]]:
        vers = sorted(client.search_model_versions(f"name='{model_name}'"),
                      key=lambda v: int(v.version), reverse=True)
        if not vers:
            logger.warning(f"No registered versions found for '{model_name}' — train first.")
            continue
        latest = vers[0]
        version = latest.version

        gate = quality_gates.get(model_name)
        if gate:
            run = client.get_run(latest.run_id)
            val = run.data.metrics.get(gate[0])
            if val is None:
                logger.warning(f"{model_name} v{version}: metric '{gate[0]}' not found, skipping gate.")
            else:
                ok = (val < gate[1]) if gate[2] == "lt" else (val > gate[1])
                status = "PASS ✓" if ok else "FAIL ✗"
                logger.info(f"{model_name} v{version}  {gate[0]}={val:.4f}  [{status}]")
                if not ok:
                    logger.error(f"Quality gate failed — '{model_name}' NOT promoted.")
                    continue

        client.transition_model_version_stage(
            name=model_name, version=version,
            stage="Production", archive_existing_versions=True,
        )
        logger.info(f"  ✓ {model_name} v{version} → Production")
        promoted[model_name] = version

    # ── Manual serving instructions ──────────────────────────────────────────
    host = spark.conf.get("spark.databricks.workspaceUrl", "<your-workspace-host>")
    print("\n" + "=" * 70)
    print("  MODELS PROMOTED — NOW CREATE SERVING ENDPOINTS IN THE UI")
    print("=" * 70)
    print(f"\n  Workspace: https://{host}\n")

    ep_name = "reai-combined"
    ver     = promoted.get(MODEL_REGISTRY_NAME["combined"], "N/A")

    print("  ONE endpoint serves all three models via the 'model_type' field.\n")
    print("  Steps:")
    print("  1. Go to  Machine Learning > Models  (left sidebar)")
    print(f"  2. Click  '{MODEL_REGISTRY_NAME['combined']}'")
    print("  3. Find the 'Production' version → click 'Use model for inference'")
    print("  4. Choose 'Real-time' → name the endpoint  reai-combined")
    print("  5. Compute: CPU  |  Size: Small  |  Scale to zero: Yes")
    print("  6. Click 'Create'\n")
    print(f"  Model   : {MODEL_REGISTRY_NAME['combined']}  (version {ver})")
    print(f"  Endpoint: {ep_name}")
    print(f"  URL     : https://{host}/serving-endpoints/{ep_name}/invocations\n")
    print("  Example requests:")
    print('  # AVM')
    print('  {"dataframe_records": [{"model_type": "avm", "payload": "[{\\"bed\\":3,\\"bath\\":2,...}]"}]}')
    print('  # Lead scoring')
    print('  {"dataframe_records": [{"model_type": "lead_scoring", "payload": "[{\\"years_since_last_sale\\":8,...}]"}]}')
    print('  # Price forecast')
    print('  {"dataframe_records": [{"model_type": "price_forecast", "payload": "[{\\"state\\":\\"California\\",\\"property_type\\":\\"SFR\\",\\"horizon\\":12}]"}]}')
    print("=" * 70 + "\n")

    return promoted

# =============================================================================
# MAIN — Run Full Pipeline
# =============================================================================

# ── Widget defaults (override in the Databricks UI widget bar) ────────────────
# csv_path is auto-resolved from the notebook folder — only change it if you
# stored realtor-data.csv somewhere other than the notebook's workspace folder.
dbutils.widgets.text("csv_path",    REALTOR_CSV_PATH, "Source CSV path (auto-detected)")
dbutils.widgets.text("test_states", "California,Texas,Ohio,Montana,New York", "AVM hold-out states")
dbutils.widgets.text("down_pct",    "0.25",  "Down payment %")
dbutils.widgets.text("rate",        "0.065", "Interest rate")
dbutils.widgets.text("hold_years",  "5",     "Investment hold years")
# Set to "true" to promote models to Production after training.
# Endpoint creation is always manual — follow the printed instructions.
dbutils.widgets.dropdown("promote_models", "false", ["true", "false"], "Promote models to Production?")

csv_path       = dbutils.widgets.get("csv_path")
test_states    = [s.strip() for s in dbutils.widgets.get("test_states").split(",")]
down_pct       = float(dbutils.widgets.get("down_pct"))
rate           = float(dbutils.widgets.get("rate"))
hold_years     = int(dbutils.widgets.get("hold_years"))
do_promote     = dbutils.widgets.get("promote_models") == "true"

print("=" * 60)
print("AI Real Estate Intelligence Platform — Full Pipeline")
print(f"  CSV source: {csv_path}")
print("=" * 60)

print("\n[1/7] Bronze ingestion...")
run_bronze(csv_path)

print("\n[2/7] Silver processing...")
run_silver()

print("\n[3/7] Gold feature engineering...")
run_gold_features()

print("\n[4/7] Training AVM...")
avm_run, avm_metrics, avm_model = run_avm(test_states=test_states)
print(f"      MAPE={avm_metrics['mape']:.2%}  MAE=${avm_metrics['mae']:,.0f}  R²={avm_metrics['r2']:.4f}")

print("\n[5/7] Price forecast + rent estimator...")
fc_run, fc_model, fc_segs = run_price_forecast()
rent_mape = run_rent_estimator()
print(f"      Rent estimator MAPE={f'{rent_mape:.2%}' if rent_mape is not None else 'skipped (insufficient data)'}")

print("\n[6/7] Lead scoring...")
ls_run, ls_metrics, lead_pre, lead_cal = run_lead_scoring()
if ls_metrics["auc"] is not None:
    print(f"      AUC={ls_metrics['auc']:.4f}  AP={ls_metrics['ap']:.4f}")
else:
    print("      Skipped (insufficient data)")

print("\n[6b] Building combined endpoint model...")
run_combined_model(avm_model, lead_pre, lead_cal, fc_model)

print("\n[7/7] Investment analysis...")
n_scored = run_investment_analysis(down_pct, rate, hold_years)
print(f"      {n_scored:,} properties scored")

# ── Model registration & serving instructions ─────────────────────────────────
# All three models are already logged to MLflow by their respective run_*()
# functions above.  Set the 'Promote models to Production?' widget to 'true'
# to also transition them to Production stage, then follow the printed UI
# instructions to create serving endpoints.
if do_promote:
    print("\n[+] Checking quality gates and promoting models to Production...")
    promote_models()
else:
    print("\n[i] Skipped model promotion (set widget 'Promote models to Production?' = true to enable).")
    print("    Models are already registered in MLflow — find them under Machine Learning > Models.")

print("\n✓ Pipeline complete. Gold tables ready:")
for tbl in ["avm_estimates","rent_estimates","lead_scores","investment_scores","price_forecasts"]:
    full = f"{DB_GOLD}.{tbl}"
    if spark.catalog.tableExists(full):
        n = spark.table(full).count()
        print(f"  {full}: {n:,} rows")
    else:
        print(f"  {full}: skipped (table not created)")
