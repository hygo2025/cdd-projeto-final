from pyspark.sql import SparkSession

from config import settings
from src.utils.enviroment import is_local
import os
from datetime import datetime, timedelta


def now_sub_days(subtract_days: int = 1) -> datetime:
    return datetime.now() - timedelta(days=subtract_days)


def _create_spark_session(name: str) -> SparkSession:
    if is_local():
        spark = (
            SparkSession.builder.appName(name)
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config("spark.executor.memory", settings.get("SPARK_EXECUTOR_MEMORY", os.environ.get("SPARK_EXECUTOR_MEMORY")))
            .config("spark.driver.memory", settings.get("SPARK_DRIVER_MEMORY", os.environ.get("SPARK_DRIVER_MEMORY")))
            .config("spark.memory.fraction", settings.get("SPARK_MEMORY_FRACTION", os.environ.get("SPARK_MEMORY_FRACTION")))
            .config("spark.driver.extraClassPath", settings.get("SPARK_DRIVER_EXTRACLASSPATH", os.environ.get("SPARK_DRIVER_EXTRACLASSPATH")))
            .config("spark.executor.extraClassPath", settings.get("SPARK_EXECUTOR_EXTRACLASSPATH", os.environ.get("SPARK_EXECUTOR_EXTRACLASSPATH")))
            .config("spark.serializer", settings.get("SPARK_SERIALIZER", os.environ.get("SPARK_SERIALIZER")))
            .config("spark.hadoop.fs.s3a.aws.profile", "default")
            .config("spark.sql.warnings", "false")
            .config("spark.sql.broadcastTimeout", "1200")
            .config("spark.sql.files.ignoreMissingFiles", "true")
            # .config("spark.eventLog.enabled", "true")
            # .config("spark.eventLog.dir", "/tmp/spark/logs")
            .getOrCreate()
        )
        return spark

    spark = SparkSession.builder.appName(name).getOrCreate()
    return spark


def create_spark_session(
        session_name: str, suffix: str = None
) -> SparkSession:
    local_name = f"{session_name}"
    if suffix:
        local_name = f"{local_name}_{suffix}"
    else:
        df_fmt = now_sub_days(1).strftime("%Y-%m-%d")
        local_name = f"{local_name}_{df_fmt}"

    return _create_spark_session(local_name)
