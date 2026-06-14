import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer

import xgboost as xgb

# to call this script:
# python model_train.py --snapshotdate "2024-09-01"


# ---------------------------------------------------------------------------
# Feature columns to use for modelling
# Numeric only — excludes Customer_ID, snapshot_date, and categoricals
# (Occupation, Credit_Mix, Payment_Behaviour, Payment_of_Min_Amount)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # attributes
    "Age",
    # financials
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    "Credit_History_Age_Months", "Num_Loan_Types",
    # clickstream
    "fe_1", "fe_2", "fe_3", "fe_4", "fe_5",
    "fe_6", "fe_7", "fe_8", "fe_9", "fe_10",
    "fe_11", "fe_12", "fe_13", "fe_14", "fe_15",
    "fe_16", "fe_17", "fe_18", "fe_19", "fe_20",
]


def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # -----------------------------------------------------------------------
    # Spark
    # -----------------------------------------------------------------------
    spark = pyspark.sql.SparkSession.builder \
        .appName("model_train") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    train_test_period_months = 18   # Jan 2023 → Jun 2024
    oot_period_months        = 2    # Jul 2024 → Aug 2024
    train_test_ratio         = 0.8

    config = {}
    config["model_train_date_str"]    = snapshotdate
    config["model_train_date"]        = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["oot_end_date"]            = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"]          = config["model_train_date"] - relativedelta(months=oot_period_months)
    config["train_test_end_date"]     = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"]   = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"]        = train_test_ratio
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"]       = oot_period_months

    print("\n--- Config ---")
    pprint.pprint(config)

    # -----------------------------------------------------------------------
    # Load label store (all partitions, then filter to window)
    # Filter on label snapshot_date (= loan_start_date + mob months)
    # -----------------------------------------------------------------------
    label_dir   = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_dir, "*.parquet"))
    assert label_files, f"No label store parquets found in {label_dir}"

    # The label store snapshot_date is when the label is observed (mob months
    # after origination). The training window is defined by label snapshot_date.
    train_start_str = config["train_test_start_date"].strftime("%Y-%m-%d")
    oot_end_str     = config["oot_end_date"].strftime("%Y-%m-%d")
    oot_start_str   = config["oot_start_date"].strftime("%Y-%m-%d")
    tt_end_str      = config["train_test_end_date"].strftime("%Y-%m-%d")

    labels_sdf = spark.read.parquet(*label_files)
    labels_sdf = labels_sdf.withColumn(
        "snapshot_date", F.col("snapshot_date").cast(StringType())
    )
    labels_sdf = labels_sdf.filter(
        (col("snapshot_date") >= train_start_str) &
        (col("snapshot_date") <= oot_end_str)
    )

    # Carry loan_start_date for joining to feature store.
    # loan_start_date = the snapshot_date in the feature store for this customer
    # (features are captured at origination, labels observed mob months later).
    if "loan_start_date" not in labels_sdf.columns:
        # Old gold label store parquets don't have loan_start_date yet.
        # Derive it: loan_start_date = snapshot_date - mob months.
        # Since mob=6, subtract 6 months. Use add_months with negative value.
        labels_sdf = labels_sdf.withColumn(
            "loan_start_date",
            F.date_format(
                F.add_months(F.to_date(col("snapshot_date"), "yyyy-MM-dd"), -6),
                "yyyy-MM-dd"
            )
        )
    else:
        labels_sdf = labels_sdf.withColumn(
            "loan_start_date", F.col("loan_start_date").cast(StringType())
        )

    print(f"\nLabel store rows in window: {labels_sdf.count()}")
    sample = labels_sdf.select("Customer_ID", "snapshot_date", "loan_start_date").limit(3).collect()
    print(f"Label sample rows: {[(r.Customer_ID, r.snapshot_date, r.loan_start_date) for r in sample]}")

    # -----------------------------------------------------------------------
    # Load feature store (all partitions, filter by loan_start_date range)
    # Features cover the same date range as loan_start_dates in the label window.
    # loan_start_date = label snapshot_date - 6 months, so feature window is
    # train_start - 6m → oot_end - 6m (but we just join on Customer_ID + date).
    # -----------------------------------------------------------------------
    feature_dir   = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_dir, "*.parquet"))
    assert feature_files, f"No feature store parquets found in {feature_dir}"

    features_sdf = spark.read.parquet(*feature_files)
    features_sdf = features_sdf.withColumn(
        "snapshot_date", F.col("snapshot_date").cast(StringType())
    )
    print(f"Feature store total rows loaded: {features_sdf.count()}")

    # -----------------------------------------------------------------------
    # Join: labels → features on Customer_ID + loan_start_date == feature snapshot_date
    # Correct join: features observed at origination, labels observed mob months later.
    # Rename feature snapshot_date to avoid column ambiguity in Spark join.
    # -----------------------------------------------------------------------
    features_renamed = features_sdf.withColumnRenamed("snapshot_date", "feature_snapshot_date")

    data_sdf = labels_sdf.join(
        features_renamed,
        on=(
            (labels_sdf["Customer_ID"] == features_renamed["Customer_ID"]) &
            (labels_sdf["loan_start_date"] == features_renamed["feature_snapshot_date"])
        ),
        how="inner"
    ).drop(features_renamed["Customer_ID"])

    data_pdf = data_sdf.toPandas()
    data_pdf["snapshot_date"] = pd.to_datetime(data_pdf["snapshot_date"])
    print(f"\nJoined dataset rows: {len(data_pdf)}")
    print(f"Label distribution:\n{data_pdf['label'].value_counts()}")

    # -----------------------------------------------------------------------
    # Split: train/test vs OOT (time-based)
    # -----------------------------------------------------------------------
    oot_start = config["oot_start_date"].date()
    oot_end   = config["oot_end_date"].date()
    tt_start  = config["train_test_start_date"].date()
    tt_end    = config["train_test_end_date"].date()

    oot_pdf        = data_pdf[(data_pdf["snapshot_date"].dt.date >= oot_start) &
                               (data_pdf["snapshot_date"].dt.date <= oot_end)]
    train_test_pdf = data_pdf[(data_pdf["snapshot_date"].dt.date >= tt_start) &
                               (data_pdf["snapshot_date"].dt.date <= tt_end)]

    # confirm all FEATURE_COLS exist in joined data
    available_cols = [c for c in FEATURE_COLS if c in data_pdf.columns]
    missing_cols   = [c for c in FEATURE_COLS if c not in data_pdf.columns]
    if missing_cols:
        print(f"WARNING: these feature cols not found and will be skipped: {missing_cols}")

    X_oot  = oot_pdf[available_cols].fillna(0)
    y_oot  = oot_pdf["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[available_cols].fillna(0),
        train_test_pdf["label"],
        test_size=1 - train_test_ratio,
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )

    print(f"\nX_train: {X_train.shape[0]} rows | default rate: {round(y_train.mean(), 3)}")
    print(f"X_test:  {X_test.shape[0]} rows  | default rate: {round(y_test.mean(), 3)}")
    print(f"X_oot:   {X_oot.shape[0]} rows   | default rate: {round(y_oot.mean(), 3)}")

    # -----------------------------------------------------------------------
    # Preprocessing: StandardScaler fitted on train only
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_oot_sc   = scaler.transform(X_oot)

    # -----------------------------------------------------------------------
    # Model 1: XGBoost
    # -----------------------------------------------------------------------
    print("\n--- Training Model 1: XGBoost ---")
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=88)
    xgb_params = {
        "n_estimators":     [25, 50, 100],
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "gamma":            [0, 0.1],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.1, 1],
        "reg_lambda":       [1, 1.5, 2],
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=xgb_params,
        scoring=make_scorer(roc_auc_score),
        n_iter=50,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train_sc, y_train)
    xgb_best = xgb_search.best_estimator_

    xgb_auc_train = roc_auc_score(y_train, xgb_best.predict_proba(X_train_sc)[:, 1])
    xgb_auc_test  = roc_auc_score(y_test,  xgb_best.predict_proba(X_test_sc)[:, 1])
    xgb_auc_oot   = roc_auc_score(y_oot,   xgb_best.predict_proba(X_oot_sc)[:, 1])

    print(f"XGBoost  | Train Gini: {round(2*xgb_auc_train-1,3)} | Test Gini: {round(2*xgb_auc_test-1,3)} | OOT Gini: {round(2*xgb_auc_oot-1,3)}")

    # -----------------------------------------------------------------------
    # Model 2: Logistic Regression
    # -----------------------------------------------------------------------
    print("\n--- Training Model 2: Logistic Regression ---")
    lr_clf = LogisticRegression(max_iter=1000, random_state=88)
    lr_params = {
        "C":       [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear", "saga"],
    }
    lr_search = RandomizedSearchCV(
        estimator=lr_clf,
        param_distributions=lr_params,
        scoring=make_scorer(roc_auc_score),
        n_iter=20,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    lr_search.fit(X_train_sc, y_train)
    lr_best = lr_search.best_estimator_

    lr_auc_train = roc_auc_score(y_train, lr_best.predict_proba(X_train_sc)[:, 1])
    lr_auc_test  = roc_auc_score(y_test,  lr_best.predict_proba(X_test_sc)[:, 1])
    lr_auc_oot   = roc_auc_score(y_oot,   lr_best.predict_proba(X_oot_sc)[:, 1])

    print(f"LogReg   | Train Gini: {round(2*lr_auc_train-1,3)} | Test Gini: {round(2*lr_auc_test-1,3)} | OOT Gini: {round(2*lr_auc_oot-1,3)}")

    # -----------------------------------------------------------------------
    # Model selection: pick best by OOT Gini
    # -----------------------------------------------------------------------
    xgb_oot_gini = round(2*xgb_auc_oot - 1, 3)
    lr_oot_gini  = round(2*lr_auc_oot  - 1, 3)

    if xgb_oot_gini >= lr_oot_gini:
        champion_name  = "xgboost"
        champion_model = xgb_best
        champion_auc   = {"train": xgb_auc_train, "test": xgb_auc_test, "oot": xgb_auc_oot}
        champion_hp    = xgb_search.best_params_
    else:
        champion_name  = "logreg"
        champion_model = lr_best
        champion_auc   = {"train": lr_auc_train, "test": lr_auc_test, "oot": lr_auc_oot}
        champion_hp    = lr_search.best_params_

    print(f"\nChampion model: {champion_name} (OOT Gini: {round(2*champion_auc['oot']-1,3)})")

    # -----------------------------------------------------------------------
    # Build model artefact (one per model, plus a champion pointer)
    # -----------------------------------------------------------------------
    def build_artefact(model_type, model, auc_dict, hp_params, search_obj):
        version = f"credit_model_{model_type}_{snapshotdate.replace('-','_')}"
        artefact = {
            "model":                     model,
            "model_type":                model_type,
            "model_version":             version,
            "is_champion":               (model_type == champion_name),
            "preprocessing_transformers": {"stdscaler": scaler},
            "feature_cols":              available_cols,
            "data_dates":                config,
            "data_stats": {
                "X_train": X_train.shape[0],
                "X_test":  X_test.shape[0],
                "X_oot":   X_oot.shape[0],
                "y_train_default_rate": round(y_train.mean(), 3),
                "y_test_default_rate":  round(y_test.mean(), 3),
                "y_oot_default_rate":   round(y_oot.mean(), 3),
            },
            "results": {
                "auc_train":  auc_dict["train"],
                "auc_test":   auc_dict["test"],
                "auc_oot":    auc_dict["oot"],
                "gini_train": round(2*auc_dict["train"]-1, 3),
                "gini_test":  round(2*auc_dict["test"]-1,  3),
                "gini_oot":   round(2*auc_dict["oot"]-1,   3),
            },
            "hp_params": hp_params,
        }
        return artefact, version

    xgb_artefact, xgb_version = build_artefact(
        "xgboost", xgb_best,
        {"train": xgb_auc_train, "test": xgb_auc_test, "oot": xgb_auc_oot},
        xgb_search.best_params_, xgb_search
    )
    lr_artefact, lr_version = build_artefact(
        "logreg", lr_best,
        {"train": lr_auc_train, "test": lr_auc_test, "oot": lr_auc_oot},
        lr_search.best_params_, lr_search
    )

    # -----------------------------------------------------------------------
    # Save both artefacts to model_bank/
    # -----------------------------------------------------------------------
    model_bank_dir = "model_bank/"
    os.makedirs(model_bank_dir, exist_ok=True)

    for artefact, version in [(xgb_artefact, xgb_version), (lr_artefact, lr_version)]:
        filepath = os.path.join(model_bank_dir, version + ".pkl")
        with open(filepath, "wb") as f:
            pickle.dump(artefact, f)
        print(f"Saved: {filepath}")

    # Write champion pointer file so inference/monitoring know which to load
    champion_version = xgb_version if champion_name == "xgboost" else lr_version
    champion_ptr = os.path.join(model_bank_dir, f"champion_{snapshotdate.replace('-','_')}.txt")
    with open(champion_ptr, "w") as f:
        f.write(champion_version + ".pkl")
    print(f"Champion pointer: {champion_ptr} -> {champion_version}.pkl")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Training Summary ===")
    print(f"XGBoost  | Train Gini: {round(2*xgb_auc_train-1,3):>6} | Test Gini: {round(2*xgb_auc_test-1,3):>6} | OOT Gini: {round(2*xgb_auc_oot-1,3):>6}")
    print(f"LogReg   | Train Gini: {round(2*lr_auc_train-1,3):>6}  | Test Gini: {round(2*lr_auc_test-1,3):>6}  | OOT Gini: {round(2*lr_auc_oot-1,3):>6}")
    print(f"Champion : {champion_name}")

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk models")
    parser.add_argument("--snapshotdate", type=str, required=True,
                        help="Model training date YYYY-MM-DD (e.g. 2024-09-01)")
    args = parser.parse_args()
    main(args.snapshotdate)
