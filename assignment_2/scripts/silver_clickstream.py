import argparse
import os
import pyspark
from utils.data_processing_silver_table import process_silver_clickstream

# Wrapper called by Airflow DAG BashOperator
# Usage: python silver_clickstream.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_clickstream") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/silver/clickstream/", exist_ok=True)

    process_silver_clickstream(
        snapshot_date_str=args.snapshotdate,
        bronze_cs_dir="datamart/bronze/clickstream/",
        silver_cs_dir="datamart/silver/clickstream/",
        spark=spark,
    )

    spark.stop()
