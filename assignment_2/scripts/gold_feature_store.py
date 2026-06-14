import argparse
import os
import pyspark
from utils.data_processing_gold_table import process_features_gold_table

# Wrapper called by Airflow DAG BashOperator
# Usage: python gold_feature_store.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/gold/feature_store/", exist_ok=True)

    process_features_gold_table(
        snapshot_date_str=args.snapshotdate,
        silver_attr_dir="datamart/silver/attributes/",
        silver_fin_dir="datamart/silver/financials/",
        silver_cs_dir="datamart/silver/clickstream/",
        gold_feature_dir="datamart/gold/feature_store/",
        spark=spark,
    )

    spark.stop()
