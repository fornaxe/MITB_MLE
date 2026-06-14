import argparse
import os
import pyspark
from utils.data_processing_silver_table import process_silver_lms

# Wrapper called by Airflow DAG BashOperator
# Usage: python silver_lms.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_lms") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/silver/lms/", exist_ok=True)

    process_silver_lms(
        snapshot_date_str=args.snapshotdate,
        bronze_lms_dir="datamart/bronze/lms/",
        silver_lms_dir="datamart/silver/lms/",
        spark=spark,
    )

    spark.stop()
