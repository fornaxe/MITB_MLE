import argparse
import os
import pyspark
from utils.data_processing_silver_table import process_silver_financials

# Wrapper called by Airflow DAG BashOperator
# Usage: python silver_financials.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_financials") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/silver/financials/", exist_ok=True)

    process_silver_financials(
        snapshot_date_str=args.snapshotdate,
        bronze_fin_dir="datamart/bronze/financials/",
        silver_fin_dir="datamart/silver/financials/",
        spark=spark,
    )

    spark.stop()
