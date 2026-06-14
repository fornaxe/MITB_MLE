import argparse
import os
import pyspark
from utils.data_processing_bronze_table import process_bronze_table

# Wrapper called by Airflow DAG BashOperator
# Usage: python bronze_lms.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_lms") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/bronze/lms/", exist_ok=True)

    process_bronze_table(
        snapshot_date_str=args.snapshotdate,
        source_path="data/lms_loan_daily.csv",
        bronze_dir="datamart/bronze/lms/",
        spark=spark,
    )

    spark.stop()
