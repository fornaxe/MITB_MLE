import argparse
import os
import glob
import pandas as pd
import numpy as np
import pickle
import pprint
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, FloatType, DateType, DoubleType

# to call this script:
# python model_monitoring.py --snapshotdate "2023-01-01" --modelname "credit_model_xgboost_2024_09_01"


def compute_psi(expected_scores, actual_scores, n_bins=10):
    """
    Compute Population Stability Index (PSI).

    Compares actual score distribution against expected (training baseline).

    PSI < 0.1  : No significant change — model is stable
    PSI 0.1–0.2: Slight shift — monitor closely
    PSI > 0.2  : Significant shift — consider retraining

    Args:
        expected_scores : array-like, scores from training period (baseline)
        actual_scores   : array-like, scores for current snapshot month
        n_bins          : number of equal-width bins (default 10)

    Returns:
        psi (float)
    """
    # define fixed bins across [0, 1]
    bins = np.linspace(0, 1, n_bins + 1)

    # compute proportions per bin
    expected_counts, _ = np.histogram(expected_scores, bins=bins)
    actual_counts,   _ = np.histogram(actual_scores,   bins=bins)

    # convert to proportions, avoid division by zero
    expected_pct = expected_counts / len(expected_scores)
    actual_pct   = actual_counts   / len(actual_scores)

    # replace zeros with small epsilon to avoid log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return round(float(psi), 6)


def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # -----------------------------------------------------------------------
    # Spark
    # -----------------------------------------------------------------------
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_monitoring") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    config = {}
    config["snapshot_date_str"]       = snapshotdate
    config["snapshot_date"]           = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["model_name"]              = modelname
    config["model_bank_directory"]    = "model_bank/"
    config["model_artefact_filepath"] = os.path.join(
        config["model_bank_directory"], modelname + ".pkl"
    )
    config["predictions_dir"]         = f"datamart/gold/model_predictions/{modelname}/"
    config["label_dir"]               = "datamart/gold/label_store/"
    config["monitoring_dir"]          = f"datamart/gold/model_monitoring/{modelname}/"

    print("\n--- Config ---")
    pprint.pprint(config)

    # -----------------------------------------------------------------------
    # Load model artefact (to get training window dates for PSI baseline)
    # -----------------------------------------------------------------------
    if not os.path.exists(config["model_artefact_filepath"]):
        print(f"Model artefact not yet available: {config['model_artefact_filepath']}")
        print("Skipping monitoring for this snapshot date.")
        spark.stop()
        return

    with open(config["model_artefact_filepath"], "rb") as f:
        model_artefact = pickle.load(f)

    train_start = model_artefact["data_dates"]["train_test_start_date"]
    train_end   = model_artefact["data_dates"]["train_test_end_date"]

    print(f"\nTraining window: {train_start.date()} → {train_end.date()}")

    # -----------------------------------------------------------------------
    # Load predictions for this snapshot date
    # -----------------------------------------------------------------------
    pred_files = glob.glob(os.path.join(config["predictions_dir"], "*.parquet"))
    if not pred_files:
        # No predictions yet — inference was skipped for this month (feature data
        # predates the store history for early backfill months). Exit cleanly.
        print(f"No prediction parquets found in {config['predictions_dir']} — skipping monitoring.")
        spark.stop()
        return

    all_pred_sdf = spark.read.parquet(*pred_files)
    all_pred_sdf = all_pred_sdf.withColumn(
        "snapshot_date", F.col("snapshot_date").cast("string")
    )

    # current month predictions
    current_pred_sdf = all_pred_sdf.filter(
        col("snapshot_date") == config["snapshot_date_str"]
    )
    current_pred_count = current_pred_sdf.count()
    if current_pred_count == 0:
        print(f"No predictions found for snapshot date {snapshotdate} — skipping monitoring.")
        spark.stop()
        return

    print(f"Predictions for {snapshotdate}: {current_pred_count} rows")

    current_pred_pdf = current_pred_sdf.toPandas()
    actual_scores    = current_pred_pdf["model_predictions"].values

    # -----------------------------------------------------------------------
    # Load label store for this snapshot date (for Gini)
    # -----------------------------------------------------------------------
    label_files = glob.glob(os.path.join(config["label_dir"], "*.parquet"))
    assert label_files, f"No label store parquets found in {config['label_dir']}"

    all_labels_sdf = spark.read.parquet(*label_files)
    all_labels_sdf = all_labels_sdf.withColumn(
        "snapshot_date", F.col("snapshot_date").cast("string")
    )

    current_labels_sdf = all_labels_sdf.filter(
        col("snapshot_date") == config["snapshot_date_str"]
    )
    current_labels_count = current_labels_sdf.count()

    print(f"Labels for {snapshotdate}: {current_labels_count} rows")

    # -----------------------------------------------------------------------
    # Compute Gini (requires labels)
    # -----------------------------------------------------------------------
    if current_labels_count > 0:
        # join predictions to labels on Customer_ID
        eval_sdf = current_pred_sdf.join(
            current_labels_sdf.select("Customer_ID", "label"),
            on="Customer_ID",
            how="inner"
        )
        eval_pdf = eval_sdf.toPandas()

        if eval_pdf["label"].nunique() < 2:
            # can't compute AUC if only one class present
            print("WARNING: Only one class in labels — Gini set to None")
            gini = None
        else:
            from sklearn.metrics import roc_auc_score
            auc  = roc_auc_score(eval_pdf["label"], eval_pdf["model_predictions"])
            gini = round(2 * auc - 1, 4)

        print(f"Gini for {snapshotdate}: {gini}")
    else:
        print(f"WARNING: No labels for {snapshotdate} — Gini set to None")
        gini = None

    # -----------------------------------------------------------------------
    # Compute PSI (compare current scores vs training baseline scores)
    # -----------------------------------------------------------------------
    # baseline = all prediction scores within the training window
    # snapshot_date is already cast to string above — compare as string
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str   = train_end.strftime("%Y-%m-%d")
    baseline_pred_sdf = all_pred_sdf.filter(
        (col("snapshot_date") >= train_start_str) &
        (col("snapshot_date") <= train_end_str)
    )
    baseline_count = baseline_pred_sdf.count()

    if baseline_count > 0:
        baseline_pdf    = baseline_pred_sdf.toPandas()
        expected_scores = baseline_pdf["model_predictions"].values
        psi             = compute_psi(expected_scores, actual_scores)
        print(f"PSI for {snapshotdate}: {psi}  (baseline rows: {baseline_count})")
    else:
        print(f"WARNING: No baseline predictions in training window — PSI set to None")
        psi = None

    # -----------------------------------------------------------------------
    # Additional monitoring stats
    # -----------------------------------------------------------------------
    mean_score    = round(float(np.mean(actual_scores)), 4)
    pct_high_risk = round(float((actual_scores > 0.5).mean() * 100), 2)

    print(f"\nMonitoring summary for {snapshotdate}:")
    print(f"  Gini          : {gini}")
    print(f"  PSI           : {psi}")
    print(f"  Mean score    : {mean_score}")
    print(f"  % high risk   : {pct_high_risk}%")

    # -----------------------------------------------------------------------
    # Save monitoring result to gold table
    # -----------------------------------------------------------------------
    os.makedirs(config["monitoring_dir"], exist_ok=True)

    result = {
        "snapshot_date":  [snapshotdate],
        "model_name":     [modelname],
        "gini":           [gini],
        "psi":            [psi],
        "mean_score":     [mean_score],
        "pct_high_risk":  [pct_high_risk],
    }
    result_pdf = pd.DataFrame(result)

    partition_name = f"{modelname}_monitoring_{snapshotdate.replace('-', '_')}.parquet"
    filepath       = config["monitoring_dir"] + partition_name

    spark.createDataFrame(result_pdf).write.mode("overwrite").parquet(filepath)
    print(f"\nSaved monitoring results to: {filepath}")

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor model performance for a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True,
                        help="Snapshot date YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                        help="Model name (without .pkl), e.g. credit_model_xgboost_2024_09_01")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
