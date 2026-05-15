import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------

spark = pyspark.sql.SparkSession.builder \
    .appName("assignment1") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_DATE = "2023-01-01"
END_DATE   = "2025-01-01"

# raw data sources
RAW_LMS        = "data/lms_loan_daily.csv"
RAW_ATTRIBUTES = "data/features_attributes.csv"
RAW_FINANCIALS = "data/features_financials.csv"
RAW_CLICKSTREAM = "data/feature_clickstream.csv"

# datamart directories
BRONZE_LMS_DIR        = "datamart/bronze/lms/"
BRONZE_ATTR_DIR       = "datamart/bronze/attributes/"
BRONZE_FIN_DIR        = "datamart/bronze/financials/"
BRONZE_CS_DIR         = "datamart/bronze/clickstream/"

SILVER_LMS_DIR        = "datamart/silver/lms/"
SILVER_ATTR_DIR       = "datamart/silver/attributes/"
SILVER_FIN_DIR        = "datamart/silver/financials/"
SILVER_CS_DIR         = "datamart/silver/clickstream/"

GOLD_LABEL_DIR        = "datamart/gold/label_store/"
GOLD_FEATURE_DIR      = "datamart/gold/feature_store/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_monthly_dates(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = datetime.strptime(end_str,   "%Y-%m-%d")
    dates = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return dates


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

make_dirs(
    BRONZE_LMS_DIR, BRONZE_ATTR_DIR, BRONZE_FIN_DIR, BRONZE_CS_DIR,
    SILVER_LMS_DIR, SILVER_ATTR_DIR, SILVER_FIN_DIR, SILVER_CS_DIR,
    GOLD_LABEL_DIR, GOLD_FEATURE_DIR
)

dates = generate_monthly_dates(START_DATE, END_DATE)
print(f"Processing {len(dates)} monthly snapshots: {dates[0]} to {dates[-1]}")


# --- Bronze ---
print("\n=== BRONZE ===")
for date_str in dates:
    bronze.process_bronze_table(date_str, RAW_LMS,        BRONZE_LMS_DIR,  spark)
    bronze.process_bronze_table(date_str, RAW_ATTRIBUTES, BRONZE_ATTR_DIR, spark)
    bronze.process_bronze_table(date_str, RAW_FINANCIALS, BRONZE_FIN_DIR,  spark)
    bronze.process_bronze_table(date_str, RAW_CLICKSTREAM, BRONZE_CS_DIR,  spark)


# --- Silver ---
print("\n=== SILVER ===")
for date_str in dates:
    silver.process_silver_lms(date_str,        BRONZE_LMS_DIR,  SILVER_LMS_DIR,  spark)
    silver.process_silver_attributes(date_str, BRONZE_ATTR_DIR, SILVER_ATTR_DIR, spark)
    silver.process_silver_financials(date_str, BRONZE_FIN_DIR,  SILVER_FIN_DIR,  spark)
    silver.process_silver_clickstream(date_str, BRONZE_CS_DIR,  SILVER_CS_DIR,   spark)


# --- Gold: label store ---
print("\n=== GOLD: label store ===")
for date_str in dates:
    gold.process_labels_gold_table(
        date_str, SILVER_LMS_DIR, GOLD_LABEL_DIR, spark, dpd=30, mob=6
    )


# --- Gold: feature store ---
# Feature store is built only at loan origination dates (one row per customer).
# We use all snapshot dates — empty partitions (no originations that month) are skipped.
print("\n=== GOLD: feature store ===")
for date_str in dates:
    gold.process_features_gold_table(
        date_str, SILVER_ATTR_DIR, SILVER_FIN_DIR, SILVER_CS_DIR, GOLD_FEATURE_DIR, spark
    )


# --- Summary ---
print("\n=== DONE ===")

label_files = glob.glob(os.path.join(GOLD_LABEL_DIR, "*.parquet"))
df_labels = spark.read.parquet(*label_files) if label_files else None

feature_files = glob.glob(os.path.join(GOLD_FEATURE_DIR, "*.parquet"))
df_features = spark.read.parquet(*feature_files) if feature_files else None

if df_labels:
    print(f"Gold label store   — total rows: {df_labels.count()}")
    df_labels.show(5)

if df_features:
    print(f"Gold feature store — total rows: {df_features.count()}")
    df_features.show(5)

spark.stop()
