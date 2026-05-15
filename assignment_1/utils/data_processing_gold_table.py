import os
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType


def process_labels_gold_table(snapshot_date_str, silver_lms_dir, gold_label_dir, spark, dpd=30, mob=6):
    """
    Build gold label store partition.
    Filter silver LMS to MOB=mob, label as 1 if DPD >= dpd threshold.
    Reused from Lab 2.
    """
    partition_name = f"silver_lms_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = silver_lms_dir + partition_name
    df = spark.read.parquet(filepath)
    print(f"[gold/label_store] loaded: {filepath} | rows: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    partition_name = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = gold_label_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[gold/label_store] saved to: {filepath} | rows: {df.count()}")

    return df


def process_features_gold_table(snapshot_date_str, silver_attr_dir, silver_fin_dir, silver_cs_dir, gold_feature_dir, spark):
    """
    Build gold feature store partition for a given snapshot date.
    Joins silver attributes + financials + clickstream on Customer_ID at that snapshot date.
    One row per customer. Clickstream missing -> nulls.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load silver partitions
    attr_file = silver_attr_dir + f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    fin_file  = silver_fin_dir  + f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    cs_file   = silver_cs_dir   + f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"

    df_attr = spark.read.parquet(attr_file)
    df_fin  = spark.read.parquet(fin_file)

    # clickstream may not exist for this snapshot date — handle gracefully
    if os.path.exists(cs_file):
        df_cs = spark.read.parquet(cs_file)
    else:
        # create empty clickstream dataframe with correct schema
        fe_cols = [f"fe_{i}" for i in range(1, 21)]
        df_cs = spark.createDataFrame([], df_attr.select("Customer_ID", "snapshot_date").schema)
        for c in fe_cols:
            df_cs = df_cs.withColumn(c, F.lit(None).cast(IntegerType()))

    print(f"[gold/feature_store] {snapshot_date_str} | attr: {df_attr.count()} | fin: {df_fin.count()} | cs: {df_cs.count()}")

    # join: attr + fin (inner — both cover all 12500 customers at all dates)
    df = df_attr.join(
        df_fin.drop("snapshot_date"),
        on="Customer_ID",
        how="inner"
    )

    # left join clickstream — preserves customers with no clickstream (fills nulls)
    df = df.join(
        df_cs.drop("snapshot_date"),
        on="Customer_ID",
        how="left"
    )

    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = gold_feature_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[gold/feature_store] saved to: {filepath} | rows: {df.count()}")

    return df
