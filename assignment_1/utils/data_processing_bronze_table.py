import os
from datetime import datetime
from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, source_path, bronze_dir, spark):
    """
    Ingest raw CSV, filter to snapshot_date, save as bronze partition. No transformation.
    Data arrives daily or monthly. Partition by snapshot_date for efficient incremental processing downstream.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    df = spark.read.csv(source_path, header=True, inferSchema=True)
    df = df.filter(col("snapshot_date") == snapshot_date)

    print(f"[bronze] {snapshot_date_str} | source: {source_path} | rows: {df.count()}")

    # derive partition filename from directory name (e.g. "lms", "attributes")
    source_name = os.path.basename(os.path.normpath(bronze_dir))
    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = bronze_dir + partition_name

    df.toPandas().to_csv(filepath, index=False)
    print(f"[bronze] saved to: {filepath}")

    return df
