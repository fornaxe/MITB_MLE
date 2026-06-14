import argparse
import os
import pyspark
from utils.data_processing_gold_table import process_labels_gold_table

# Wrapper called by Airflow DAG BashOperator
# Usage: python gold_label_store.py --snapshotdate "2023-01-01"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_label_store") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    os.makedirs("datamart/gold/label_store/", exist_ok=True)

    process_labels_gold_table(
        snapshot_date_str=args.snapshotdate,
        silver_lms_dir="datamart/silver/lms/",
        gold_label_dir="datamart/gold/label_store/",
        spark=spark,
        dpd=30,
        mob=6,
    )

    spark.stop()
