import os
import re
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ---------------------------------------------------------------------------
# LMS (label store source) — reused from Lab 2
# ---------------------------------------------------------------------------

def process_silver_lms(snapshot_date_str, bronze_lms_dir, silver_lms_dir, spark):
    """
    Clean and augment LMS bronze partition.
    Computes MOB (month on book) and DPD (days past due).
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    partition_name = f"bronze_lms_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = bronze_lms_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver/lms] loaded: {filepath} | rows: {df.count()}")

    # enforce schema
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # mob = installment_num
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # dpd: days past due derived from overdue amount
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    partition_name = f"silver_lms_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = silver_lms_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver/lms] saved to: {filepath}")

    return df


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def process_silver_attributes(snapshot_date_str, bronze_attr_dir, silver_attr_dir, spark):
    """
    Clean attributes bronze partition.
    - Drop PII (Name, SSN)
    - Clean Age: strip trailing underscore, cast to int, nullify implausible values
    - Clean Occupation: map garbage placeholder to null
    """
    partition_name = f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = bronze_attr_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver/attributes] loaded: {filepath} | rows: {df.count()}")

    # drop PII
    df = df.drop("Name", "SSN")

    # clean Age: strip trailing underscore then cast; nullify outside [15, 100]
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), r"_+$", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") >= 15) & (col("Age") <= 100), col("Age")).otherwise(None))

    # clean Occupation: blank placeholder -> null
    df = df.withColumn("Occupation", F.when(col("Occupation") == "_______", None).otherwise(col("Occupation")))

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    partition_name = f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = silver_attr_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver/attributes] saved to: {filepath}")

    return df


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

def _parse_credit_history_months(val):
    """Parse 'X Years and Y Months' -> total integer months. Returns None if unparseable."""
    if val is None:
        return None
    match = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(val))
    if match:
        return int(match.group(1)) * 12 + int(match.group(2))
    return None

_parse_credit_history_months_udf = udf(_parse_credit_history_months, IntegerType())


def process_silver_financials(snapshot_date_str, bronze_fin_dir, silver_fin_dir, spark):
    """
    Clean financials bronze partition.
    - Strip trailing underscores from numeric columns, cast to correct types
    - Nullify sentinel/garbage values
    - Parse Credit_History_Age string to integer months
    - Encode Type_of_Loan as count of loan types
    - Nullify garbage categoricals
    """
    partition_name = f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = bronze_fin_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver/financials] loaded: {filepath} | rows: {df.count()}")

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # --- numeric columns with trailing underscore ---
    strip_cols_float = ["Annual_Income", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"]
    for c in strip_cols_float:
        df = df.withColumn(c, F.regexp_replace(col(c).cast(StringType()), r"[_\s]+$", ""))
        df = df.withColumn(c, col(c).cast(FloatType()))

    strip_cols_int = ["Num_of_Loan", "Num_of_Delayed_Payment"]
    for c in strip_cols_int:
        df = df.withColumn(c, F.regexp_replace(col(c).cast(StringType()), r"[_\s]+$", ""))
        df = df.withColumn(c, col(c).cast(FloatType()).cast(IntegerType()))

    # nullify sentinel values
    df = df.withColumn("Num_of_Loan", F.when(col("Num_of_Loan") < 0, None).otherwise(col("Num_of_Loan")))
    df = df.withColumn("Num_of_Delayed_Payment", F.when(col("Num_of_Delayed_Payment") < 0, None).otherwise(col("Num_of_Delayed_Payment")))

    # Changed_Credit_Limit: underscore placeholder -> null
    df = df.withColumn("Changed_Credit_Limit",
        F.regexp_replace(col("Changed_Credit_Limit").cast(StringType()), r"^_+$", ""))
    df = df.withColumn("Changed_Credit_Limit",
        F.when(col("Changed_Credit_Limit") == "", None)
         .otherwise(col("Changed_Credit_Limit").cast(FloatType())))

    # Amount_invested_monthly: __10000__ -> null (already handled by strip above giving non-numeric)
    # After strip+cast, non-numeric becomes null automatically via cast — already done above

    # Monthly_Balance extreme outlier -> null (cast handles it, value is non-numeric after strip)

    # Credit_Mix: underscore placeholder -> null
    df = df.withColumn("Credit_Mix",
        F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))

    # Payment_of_Min_Amount: keep Yes/No/NM as-is (NM = not mentioned, valid category)

    # Payment_Behaviour: garbage -> null
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour")))

    # Credit_History_Age: parse "X Years and Y Months" -> integer months
    df = df.withColumn("Credit_History_Age_Months", _parse_credit_history_months_udf(col("Credit_History_Age")))
    df = df.drop("Credit_History_Age")

    # Type_of_Loan: count comma-separated entries; null -> 0
    df = df.withColumn("Num_Loan_Types",
        F.when(col("Type_of_Loan").isNull(), F.lit(0))
         .otherwise(
             F.size(F.split(col("Type_of_Loan"), ","))
         ).cast(IntegerType()))
    df = df.drop("Type_of_Loan")

    partition_name = f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = silver_fin_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver/financials] saved to: {filepath}")

    return df


# ---------------------------------------------------------------------------
# Clickstream
# ---------------------------------------------------------------------------

def process_silver_clickstream(snapshot_date_str, bronze_cs_dir, silver_cs_dir, spark):
    """
    Clean clickstream bronze partition.
    Data is already clean integers — just enforce types and save.
    Negative values are valid signal.
    """
    partition_name = f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = bronze_cs_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[silver/clickstream] loaded: {filepath} | rows: {df.count()}")

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # fe columns are already int — just ensure correct type
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in fe_cols:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    partition_name = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = silver_cs_dir + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f"[silver/clickstream] saved to: {filepath}")

    return df
