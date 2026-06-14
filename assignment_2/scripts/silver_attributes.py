import argparse
import os
import pyspark
from utils.data_processing_silver_table import process_silver_attributes

# Wrapper called by Airflow DAG BashOperator
# Usage: python silver_attributes.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_attributes") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/silver/attributes/", exist_ok=True)

    process_silver_attributes(
        snapshot_date_str=args.snapshotdate,
        bronze_attr_dir="datamart/bronze/attributes/",
        silver_attr_dir="datamart/silver/attributes/",
        spark=spark,
    )

    spark.stop()
