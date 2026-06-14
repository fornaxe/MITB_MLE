from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Model training is a one-off event: only runs on the designated training date.
# All other months skip the training tasks and go straight to inference.
# ---------------------------------------------------------------------------
MODEL_TRAIN_DATE = "2024-09-01"

# Both models trained on the same training date — referenced by name in
# inference + monitoring so the DAG knows which pkl files to load.
CHAMPION_MODEL = "credit_model_xgboost_2024_09_01"
CHALLENGER_MODEL = "credit_model_logreg_2024_09_01"

SCRIPTS_DIR = "/opt/airflow/scripts"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "credit_risk_pipeline",
    default_args=default_args,
    description="End-to-end credit risk ML pipeline — monthly backfill Jan 2023 to Jan 2025",
    schedule_interval="0 0 1 * *",   # 1st of every month at midnight
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,
) as dag:

    # -----------------------------------------------------------------------
    # 0. Dependency check — placeholder (IRL: FileSensor / ExternalTaskSensor)
    # -----------------------------------------------------------------------
    dep_check = DummyOperator(task_id="dep_check")


    # -----------------------------------------------------------------------
    # 1. Bronze layer — ingest raw CSVs, filter to snapshot_date
    # -----------------------------------------------------------------------
    bronze_lms = BashOperator(
        task_id="bronze_lms",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 bronze_lms.py --snapshotdate "{{ ds }}"'
        ),
    )

    bronze_attributes = BashOperator(
        task_id="bronze_attributes",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 bronze_attributes.py --snapshotdate "{{ ds }}"'
        ),
    )

    bronze_financials = BashOperator(
        task_id="bronze_financials",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 bronze_financials.py --snapshotdate "{{ ds }}"'
        ),
    )

    bronze_clickstream = BashOperator(
        task_id="bronze_clickstream",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 bronze_clickstream.py --snapshotdate "{{ ds }}"'
        ),
    )


    # -----------------------------------------------------------------------
    # 2. Silver layer — clean + type-cast each source
    # -----------------------------------------------------------------------
    silver_lms = BashOperator(
        task_id="silver_lms",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 silver_lms.py --snapshotdate "{{ ds }}"'
        ),
    )

    silver_attributes = BashOperator(
        task_id="silver_attributes",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 silver_attributes.py --snapshotdate "{{ ds }}"'
        ),
    )

    silver_financials = BashOperator(
        task_id="silver_financials",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 silver_financials.py --snapshotdate "{{ ds }}"'
        ),
    )

    silver_clickstream = BashOperator(
        task_id="silver_clickstream",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 silver_clickstream.py --snapshotdate "{{ ds }}"'
        ),
    )


    # -----------------------------------------------------------------------
    # 3. Gold layer — label store + feature store
    # -----------------------------------------------------------------------
    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 gold_label_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 gold_feature_store.py --snapshotdate "{{ ds }}"'
        ),
    )

    stores_completed = DummyOperator(task_id="stores_completed")


    # -----------------------------------------------------------------------
    # 4. Model training — only fires on MODEL_TRAIN_DATE
    #    ShortCircuitOperator: if condition is False, downstream tasks are
    #    SKIPPED (not failed), so the DAG run still completes green.
    # -----------------------------------------------------------------------
    def _is_training_date(**context):
        """Return True only on the designated model training date."""
        return context["ds"] == MODEL_TRAIN_DATE

    should_train = ShortCircuitOperator(
        task_id="should_train",
        python_callable=_is_training_date,
        ignore_downstream_trigger_rules=False,  # only skip the training branch
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 model_train.py --snapshotdate "{{ ds }}"'
        ),
    )

    training_completed = DummyOperator(
        task_id="training_completed",
        trigger_rule="none_failed_min_one_success",
    )


    # -----------------------------------------------------------------------
    # 5. Model inference — both models, every month
    #    trigger_rule=all_done: runs even when training was skipped
    # -----------------------------------------------------------------------
    inference_start = DummyOperator(
        task_id="inference_start",
        trigger_rule="all_done",   # proceed whether training ran or was skipped
    )

    model_champion_inference = BashOperator(
        task_id="model_champion_inference",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            f'python3 model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname "{CHAMPION_MODEL}"'
        ),
    )

    model_challenger_inference = BashOperator(
        task_id="model_challenger_inference",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            f'python3 model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'
        ),
    )

    inference_completed = DummyOperator(task_id="inference_completed")


    # -----------------------------------------------------------------------
    # 6. Model monitoring — Gini + PSI per model, every month
    # -----------------------------------------------------------------------
    monitor_start = DummyOperator(task_id="monitor_start")

    model_champion_monitor = BashOperator(
        task_id="model_champion_monitor",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            f'python3 model_monitoring.py --snapshotdate "{{{{ ds }}}}" --modelname "{CHAMPION_MODEL}"'
        ),
    )

    model_challenger_monitor = BashOperator(
        task_id="model_challenger_monitor",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            f'python3 model_monitoring.py --snapshotdate "{{{{ ds }}}}" --modelname "{CHALLENGER_MODEL}"'
        ),
    )

    monitor_completed = DummyOperator(task_id="monitor_completed")


    # -----------------------------------------------------------------------
    # 7. Visualise — reads all monitoring results, saves charts
    # -----------------------------------------------------------------------
    visualise = BashOperator(
        task_id="visualise",
        bash_command=(
            f"cd {SCRIPTS_DIR} && "
            'python3 visualise.py --snapshotdate "{{ ds }}"'
        ),
    )

    pipeline_completed = DummyOperator(task_id="pipeline_completed")


    # -----------------------------------------------------------------------
    # Task dependencies
    # -----------------------------------------------------------------------

    # dep_check → all bronze (parallel)
    dep_check >> [bronze_lms, bronze_attributes, bronze_financials, bronze_clickstream]

    # bronze → silver (each source independently)
    bronze_lms        >> silver_lms
    bronze_attributes >> silver_attributes
    bronze_financials >> silver_financials
    bronze_clickstream >> silver_clickstream

    # silver → gold (label needs lms only; features needs attr+fin+cs)
    silver_lms                                              >> gold_label_store
    [silver_attributes, silver_financials, silver_clickstream] >> gold_feature_store

    # both gold stores must complete before moving on
    [gold_label_store, gold_feature_store] >> stores_completed

    # stores → training gate (only runs on MODEL_TRAIN_DATE)
    stores_completed >> should_train >> model_train >> training_completed

    # stores + training gate → inference (runs every month via all_done)
    stores_completed  >> inference_start
    training_completed >> inference_start

    # inference: both models in parallel
    inference_start >> [model_champion_inference, model_challenger_inference] >> inference_completed

    # inference → monitoring: both models in parallel
    inference_completed >> monitor_start
    monitor_start >> [model_champion_monitor, model_challenger_monitor] >> monitor_completed

    # monitoring → visualise → done
    monitor_completed >> visualise >> pipeline_completed
