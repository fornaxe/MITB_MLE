import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, FloatType, DateType

# to call this script:
# python model_inference.py --snapshotdate "2023-01-01" --modelname "credit_model_xgboost_2024_09_01"


def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # -----------------------------------------------------------------------
    # Spark
    # -----------------------------------------------------------------------
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_inference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = {}
    config["snapshot_date_str"]      = snapshotdate
    config["snapshot_date"]          = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["model_name"]             = modelname
    config["model_bank_directory"]   = "model_bank/"
    config["model_artefact_filepath"] = os.path.join(
        config["model_bank_directory"], modelname + ".pkl"
    )

    print("\n--- Config ---")
    pprint.pprint(config)

    # -----------------------------------------------------------------------
    # Load model artefact
    # -----------------------------------------------------------------------
    if not os.path.exists(config["model_artefact_filepath"]):
        # Model not yet trained — this happens during backfill for months
        # before the training date (2024-09-01). Exit cleanly so Airflow
        # marks the task as success rather than failing.
        print(f"Model artefact not yet available: {config['model_artefact_filepath']}")
        print("Skipping inference for this snapshot date.")
        spark.stop()
        return

    with open(config["model_artefact_filepath"], "rb") as f:
        model_artefact = pickle.load(f)

    print(f"Model loaded: {modelname}")
    print(f"  Model type    : {model_artefact.get('model_type', 'unknown')}")
    print(f"  Is champion   : {model_artefact.get('is_champion', 'unknown')}")
    print(f"  OOT Gini      : {model_artefact['results']['gini_oot']}")
    print(f"  Feature cols  : {len(model_artefact['feature_cols'])} features")

    # -----------------------------------------------------------------------
    # Load feature store for this snapshot date
    # Features are observed at loan origination (loan_start_date = snapshot_date - 6m).
    # The inference snapshot_date is the label observation date (mob=6 months later).
    # -----------------------------------------------------------------------
    feature_snapshot_date = config["snapshot_date"] - relativedelta(months=6)
    feature_snapshot_str  = feature_snapshot_date.strftime("%Y-%m-%d")
    print(f"Feature lookup date (loan_start_date = snapshot_date - 6m): {feature_snapshot_str}")

    feature_dir   = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_dir, "*.parquet"))
    assert feature_files, f"No feature store parquets found in {feature_dir}"

    features_sdf = spark.read.parquet(*feature_files)
    features_sdf = features_sdf.withColumn(
        "snapshot_date", F.col("snapshot_date").cast("string")
    )
    features_sdf = features_sdf.filter(
        col("snapshot_date") == feature_snapshot_str
    )

    row_count = features_sdf.count()
    print(f"\nFeature store rows for {feature_snapshot_str}: {row_count}")
    if row_count == 0:
        # Feature data doesn't exist for this origination date — happens for
        # the first mob months of the backfill where loan_start_date predates
        # the feature store history. Exit cleanly.
        print(f"No feature data for origination date {feature_snapshot_str} — skipping inference.")
        spark.stop()
        return

    features_pdf = features_sdf.toPandas()

    # -----------------------------------------------------------------------
    # Prepare features — use same feature_cols saved in artefact
    # -----------------------------------------------------------------------
    feature_cols = model_artefact["feature_cols"]

    # check all expected cols are present
    missing = [c for c in feature_cols if c not in features_pdf.columns]
    if missing:
        print(f"WARNING: missing feature cols filled with 0: {missing}")

    X_inference = features_pdf.reindex(columns=feature_cols, fill_value=0).fillna(0)

    # apply StandardScaler saved in artefact (fitted on train only)
    scaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference_sc = scaler.transform(X_inference)

    print(f"X_inference shape: {X_inference_sc.shape}")

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------
    model = model_artefact["model"]
    scores = model.predict_proba(X_inference_sc)[:, 1]

    # -----------------------------------------------------------------------
    # Build output dataframe
    # Store the label snapshot_date (inference date), not the feature date,
    # so monitoring can filter predictions by the correct observation month.
    # -----------------------------------------------------------------------
    output_pdf = features_pdf[["Customer_ID"]].copy()
    output_pdf["snapshot_date"]      = snapshotdate   # label observation date
    output_pdf["model_name"]         = modelname
    output_pdf["model_predictions"]  = scores

    print(f"\nScore distribution:")
    print(f"  Mean  : {round(scores.mean(), 4)}")
    print(f"  Median: {round(np.median(scores), 4)}")
    print(f"  Min   : {round(scores.min(), 4)}")
    print(f"  Max   : {round(scores.max(), 4)}")
    print(f"  % > 0.5 (high risk): {round((scores > 0.5).mean() * 100, 1)}%")

    # -----------------------------------------------------------------------
    # Save to gold model_predictions table
    # -----------------------------------------------------------------------
    gold_dir = f"datamart/gold/model_predictions/{modelname}/"
    os.makedirs(gold_dir, exist_ok=True)

    partition_name = f"{modelname}_predictions_{snapshotdate.replace('-', '_')}.parquet"
    filepath = gold_dir + partition_name

    spark.createDataFrame(output_pdf).write.mode("overwrite").parquet(filepath)
    print(f"\nSaved predictions to: {filepath}")

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True,
                        help="Snapshot date YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                        help="Model name (without .pkl), e.g. credit_model_xgboost_2024_09_01")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
