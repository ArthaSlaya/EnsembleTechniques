# 1) Check which expected z-cols exist
z_cols = {
    "SC": "SC_z_score_session_cnt",
    "SS": "SS_z_score_max",
    "SL": "SL_z_score",
    "ZB": "ZB_z_score",
    "BU": "BU_z_score_bytes_usage",
    "IDLE":"IDLE_z_idle",
}

missing = [c for c in z_cols.values() if c not in df_feat.columns]
print("Missing z-cols:", missing)

# Are there any NaNs / all zeros in those cols?
for k, col in z_cols.items():
    if col in df_feat.columns:
        s = pd.to_numeric(df_feat[col], errors="coerce")
        print(f"{col}: n={len(s)}, nonnull={s.notna().sum()}, "
              f"all_zero={bool((s.fillna(0)==0).all())}, "
              f"mean={s.mean():.3f}, std={s.std():.3f}")


# ============================================================================
# dip-ingestion-platform/mod-ml/aaa-inferencing-lambda/lambda_function.py
# ============================================================================
import base64
import json
import logging

import boto3
from pandas import DataFrame
from splunk_hec_handler import SplunkHecHandler

from lib.app_config import AppConfig
from lib import process_anomaly


# This filter excludes logs from boto3 and botocore
class ExcludeBotoLogsFilter(logging.Filter):
    def filter(self, record):
        # Exclude logs from boto3 and botocore
        return not (
            record.name.startswith("boto3") or record.name.startswith("botocore")
        )


def setup_logging(config: AppConfig) -> logging.Logger:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    if config.splunk_host and config.splunk_token:
        try:
            splunk_handler = SplunkHecHandler(
                host=config.splunk_host,
                token=config.splunk_token,
                port=config.splunk_port,
                proto=config.splunk_proto,
                index=config.splunk_index or "main",
                source=config.splunk_source,
                sourcetype=config.splunk_sourcetype,
                ssl_verify=False,
                batch_size_count=20,
            )
            splunk_handler.addFilter(ExcludeBotoLogsFilter())
            splunk_handler.setLevel(logging.INFO)
            root_logger.addHandler(splunk_handler)
        except Exception as e:
            root_logger.error(
                f"Failed to initialize Splunk HEC handler: {e}", exc_info=True
            )
    return root_logger


# =================================================================================
# Global Initializations (Lambda Cold Start)
# =================================================================================
try:
    CONFIG = AppConfig.from_env()
    LOGGER = setup_logging(CONFIG)
    KINESIS_CLIENT = boto3.client("kinesis")
    SAGEMAKER_CLIENT = boto3.client("sagemaker")
    LOADED_MODELS = {}
except Exception as e:
    initialization_logger = logging.getLogger()
    initialization_logger.setLevel(logging.INFO)
    initialization_logger.error(f"CRITICAL: Failed during Lambda initialization: {e}")
    CONFIG = None


def ensure_splunk_handler(config: AppConfig, logger: logging.Logger):
    if (
        not any(isinstance(h, SplunkHecHandler) for h in logger.handlers)
        and config.splunk_host
        and config.splunk_token
    ):
        try:
            splunk_handler = SplunkHecHandler(
                host=config.splunk_host,
                token=config.splunk_token,
                port=config.splunk_port,
                proto=config.splunk_proto,
                index=config.splunk_index,
                source=config.splunk_source,
                sourcetype=config.splunk_sourcetype,
                ssl_verify=False,
                batch_size=20,
                flush_interval=0,
                queue_size=0,
                run_async=False,
            )
            splunk_handler.setLevel(logging.INFO)
            logger.addHandler(splunk_handler)
        except Exception as e:
            logger.error(
                f"Handler Failed to re-attach SplunkHecHandler: {e}", exc_info=True
            )


def lambda_handler(event, context):

    if CONFIG is None:
        LOGGER.error(
            "CRITICAL: Lambda initialization failed. "
            "Please check the environment variables and configuration."
        )
        # Return an error response to indicate that the Lambda function failed to initialize
        return {
            "statusCode": 500,
            "body": "Internal Server Error: Lambda initialization failed.",
        }
    ensure_splunk_handler(CONFIG, LOGGER)

    data = [
        json.loads(base64.b64decode(record["kinesis"]["data"]).decode("utf-8"))
        for record in event["Records"]
    ]
    df = DataFrame(data)
    if "app_id" not in df.columns:
        LOGGER.error("Input data is missing required 'app_id' column.")
        return {
            "statusCode": 400,
            "body": "Input data is missing required 'app_id' column.",
        }

    else:
        LOGGER.info(f"number of records read in: {len(df)}")
        try:
            process_anomaly.run_inferencing(loaded_models=LOADED_MODELS, df=df)
            return {
                "statusCode": 200,
                "body": "Successfully processed {} records.".format(df.shape[0]),
            }
        except Exception as e:
            LOGGER.error(
                f"Error in process_anomaly.run_inferencing: {e}", exc_info=True
            )
            return {
                "statusCode": 500,
                "body": f"Internal Server Error: {str(e)}",
            }

# ============================================================================
# dip-ingestion-platform/mod-ml/aaa-inferencing-lambda/lib/process_anomaly.py
# ============================================================================
from io import BytesIO
import joblib
from os import environ
import aws_utils_lambda
import pandas as pd
from pandas import DataFrame
import numpy as np
import boto3
import json
from sklearn.ensemble import IsolationForest
import time
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = True


def create_anomaly_fields(df: DataFrame, cause: str) -> DataFrame:
    # Map cause to column names
    df["zero_bytes_normal_count"] = np.where(
        df["rolling_mean_zero_bytes_Count"] > 5, 5, df["rolling_mean_zero_bytes_Count"]
    )

    column_map = {
        "zeroBytes": (
            "max_zero_bytes_Count",
            "ZerobytesUsagenAnomaly",
            "zero_bytes_normal_count",
        ),
        "shortSession": (
            "max_shortLength",
            "shortSessionAnomaly",
            "expanding_mean_max_shortLength",
        ),
    }
    column_count, column_anomaly, column_normal = column_map[cause]

    # normal session counts for particular cause
    df[f"NormalSessionCount_{cause}"] = np.where(
        df[column_anomaly] == 0, df[column_count], np.nan
    )

    df[f"NormalSessionCount_{cause}"] = (
        df[f"NormalSessionCount_{cause}"].fillna(df[column_normal])
    ).astype(int)

    df[f"AnomalySessionCount_{cause}"] = np.where(
        df[column_anomaly] == 1,
        df[column_count],
        df["AnomalySessionCount"],
    ).astype(int)

    anomaly_count = df[column_anomaly].sum()
    log.debug(f"{anomaly_count} anomalies detected for cause: {cause}")

    return df


def create_output_df(df: DataFrame) -> DataFrame:
    # add anomaly cause field:
    df["ShortSessionLabel"] = np.where(
        df["shortSessionAnomaly"] == 1, "ShortSession", ""
    )
    df["ZeroBytesLabel"] = np.where(
        df["ZerobytesUsagenAnomaly"] == 1, "ZeroByteSession", ""
    )

    # Concatenate the labels with a comma separator when both conditions are true
    df["AnomalyCause"] = (
        df["ShortSessionLabel"]
        + np.where(
            (df["ShortSessionLabel"] != "") & (df["ZeroBytesLabel"] != ""), ", ", ""
        )
        + df["ZeroBytesLabel"]
    )
    df.drop(["ShortSessionLabel", "ZeroBytesLabel"], axis=1, inplace=True)

    for cause in ("zeroBytes", "shortSession"):
        df = create_anomaly_fields(df, cause)

    # add additional field
    df["DurationInMin"] = 60
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["AnomalyEventTime"] = (
        df["date"] + pd.to_timedelta(df["hour"], unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    num_missing = (df["AnomalyCause"] == "").sum()
    log.info(f"{num_missing} rows have no anomaly cause assigned") 

    num_with_cause = (df["AnomalyCause"] != "").sum()
    log.info(f"{num_with_cause} rows have anomaly causes assigned")

    return df


def create_anomaly_df(df: DataFrame) -> dict:
    df = create_output_df(df)
    # get the anomaly rows
    df_anomaly = df[(df["anomaly"] == 1) & (df["AnomalyCause"] != "")]
    log.info(f"{len(df_anomaly)} anomaly rows retained for processing (non-empty AnomalyCause)")

    df_anomaly = df_anomaly[
        [
            "app_id",
            "device_id",
            "AnomalyCause",
            "NormalSessionCount_zeroBytes",
            "AnomalySessionCount_zeroBytes",
            "NormalSessionCount_shortSession",
            "AnomalySessionCount_shortSession",
            "DurationInMin",
            "AvgSessionLengthTime",
            "AnomalyEventTime",
            "anomaly",
        ]
    ]

    df_anomaly["AnomalyCause"] = df_anomaly["AnomalyCause"].str.split(", ")

    # Split the rows only where multiple causes are present
    df_anomaly = df_anomaly.explode("AnomalyCause")

    # Mapping the columns based on causes
    conditions = [
        df_anomaly["AnomalyCause"] == "ShortSession",
        df_anomaly["AnomalyCause"] == "ZeroByteSession",
    ]

    choices_normal = [
        df_anomaly["NormalSessionCount_shortSession"],
        df_anomaly["NormalSessionCount_zeroBytes"],
    ]

    choices_anomaly = [
        df_anomaly[
            "AnomalySessionCount_shortSession"
        ],  # Fixed typo in the original code
        df_anomaly["AnomalySessionCount_zeroBytes"],
    ]

    # Apply vectorized mapping
    df_anomaly["NormalSessionCount"] = np.select(
        conditions, choices_normal, default=None
    )
    df_anomaly["AnomalySessionCount"] = np.select(
        conditions, choices_anomaly, default=None
    )

    # Drop the original columns
    df_anomaly.drop(
        [
            "NormalSessionCount_zeroBytes",
            "AnomalySessionCount_zeroBytes",
            "NormalSessionCount_shortSession",
            "AnomalySessionCount_shortSession",
            "anomaly",
        ],
        axis=1,
        inplace=True,
    )

    return df_anomaly


def send_to_kinesis(records, stream_name, region):

    log.info(f"Sending {len(records)} records to Kinesis stream: {stream_name} in region: {region}")

    # Create a Kinesis client
    kinesis_client = boto3.client("kinesis", region_name=region)

    # Convert the string data to bytes
    bytes_data = records.encode("utf-8")
    try:
        kinesis_client.put_record(
            StreamName=stream_name,
            Data=bytes_data,
            PartitionKey="DeviceId",  # A key used to distribute data across shards
        )
    except Exception as e:
        log.error(f"Failed to put record in the kinesis stream: {e}", exc_info=True)
        raise  # Re-raise to propagate error to Lambda


def process_and_send_records(records, stream_name, region):
    if records:
        records = sorted(
            records, key=lambda x: (x.get("DeviceId"), x.get("AnomalyEventTime"))
        )
        records = {"events": records}
        records = json.dumps(records)

        #log.debug(f"Sending JSON batch of {len(json.loads(records)['events'])} records to Kinesis")
        send_to_kinesis(records, stream_name, region)


def batch_records(records, batch_size):
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def send_records_to_kinesis(df: DataFrame, stream_name, region):

    # get the correct columns for each anomaly cause
    df.rename(columns={"device_id": "DeviceId"}, inplace=True)
    df["AvgSessionLengthTime"] = df["AvgSessionLengthTime"].fillna(0)
    df["AvgSessionLengthTime"] = df["AvgSessionLengthTime"].astype(int)
    df["AppId"] = df["app_id"].astype(int)
    df["AnomalyEventTime"] = df["AnomalyEventTime"].astype(str)
    df_sent = df[
        [
            "DeviceId",
            "AppId",
            "AnomalyCause",
            "NormalSessionCount",
            "AnomalySessionCount",
            "DurationInMin",
            "AvgSessionLengthTime",
            "AnomalyEventTime",
        ]
    ]

    # Convert DataFrame to list of dictionaries
    records = df_sent.to_dict("records")
    log.info(f"Preparing to send {len(records)} records to Kinesis stream: {stream_name} in region: {region}")

    # Process records
    for record in records:
        if record["AnomalyCause"] == "ZeroByteSession":
            record.pop("AvgSessionLengthTime", None)

    # Send records in batches and each batch contains 1000 records
    for batch in batch_records(records, 1000):
        log.debug(f"Sending batch of {len(batch)} records to Kinesis")
        process_and_send_records(batch, stream_name, region)
        time.sleep(1)


def load_model_from_s3(
    loaded_models: dict,
    prefix_model: str,
    pod_name_model: str,
    region_model: str,
    app_id: str,
) -> IsolationForest:

    if app_id in loaded_models:
        log.debug(f"Model for app_id={app_id} loaded from cache") #1
        return loaded_models[app_id]
    else:
        # Create a boto3 client
        s3_client = boto3.client("s3")
        try:
            # Load the model from the buffer using joblib
            with aws_utils_lambda.get_aurora_postgresql_connection(
                prefix_model, pod_name_model, region_model
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"SELECT model_path FROM ip_{pod_name_model}.aaa_model_path "
                        f"WHERE app_id={app_id};"
                    )
                    row = cursor.fetchone()
                    # Create a cloud watch alert for this error
                    if row is None:
                        log.error(
                            f"No model_path found for app_id={app_id} in ip_{pod_name_model}.aaa_model_path" #2
                        )
                        return None
                    bucket_name, model_key = aws_utils_lambda.split_s3_path(row[0])
                    # # Create a buffer
                    model_buffer = BytesIO()
                    # # Download the model file from S3 to the buffer
                    s3_client.download_fileobj(bucket_name, model_key, model_buffer)
                    #
                    # # Set the buffer's pointer to the beginning
                    model_buffer.seek(0)

                    model = joblib.load(model_buffer)
                    log.info(f"Successfully loaded model for app_id={app_id} from {bucket_name}/{model_key}") #3

                    loaded_models[app_id] = model

            return model
        except Exception as e:
            # Create a cloud watch alert for this error
            log.error(f"Error accessing model table for app_id={app_id}: {e}") #1 #4
            return None


def run_inferencing(loaded_models: dict, df: DataFrame) -> None:
    log.info(f"Starting inferencing for {len(df)} rows across {df['app_id'].nunique()} app_ids")
    prefix = environ.get("PREFIX")
    pod_name = environ.get("POD_NAME")
    region = environ.get("REGION_NAME")
    df_output = pd.DataFrame()
    log.info(f"Unique app_ids to process: {df['app_id'].unique().tolist()}")
    for app_id in df["app_id"].unique():
        
        try:
            df_app_id = df[df["app_id"] == app_id].copy()
            log.info(f"Processing app_id={app_id}, records={len(df_app_id)}, devices={df_app_id['device_id'].nunique()}")
            df_app_id.sort_values(
                by=["app_id", "device_id", "date"], inplace=True, ignore_index=True
            )

            log.info(
                f"app_id={app_id} flags: shortSessionAnomaly={df_app_id['shortSessionAnomaly'].sum()}, "
                 f"ZerobytesUsagenAnomaly={df_app_id['ZerobytesUsagenAnomaly'].sum()}"
                )

            X = df_app_id[
                [
                    "session_count",
                    "expanding_mean_session_count",
                    "AnomalySessionCount",
                    "max_shortLength",
                    "expanding_mean_max_shortLength",
                    "shortSessionAnomaly",
                    "session_time",
                    "expanding_mean_session_time",
                    "sessionTimeAnomalyBasedOnHistory",
                    "max_zero_bytes_Count",
                    "rolling_mean_zero_bytes_Count",
                    "ZerobytesUsagenAnomaly",
                ]
            ]
            X = X.astype(float)
            X = X.fillna(X.mean())
            X = X.fillna(0)

            model = load_model_from_s3(
                loaded_models=loaded_models,
                prefix_model=prefix,
                pod_name_model=pod_name,
                region_model=region,
                app_id=app_id,
            )
            
            if model:
                df_app_id["anomaly"] = model.predict(X)

                # do the mapping for IsolationForest model to 0 and 1 values, where 1 is Anomaly

                df_app_id["anomaly"] = df_app_id["anomaly"].apply(
                    lambda x: 1 if x == -1 else 0
                )
                df_anomalies: DataFrame = create_anomaly_df(df_app_id)

                if len(df_anomalies) > 0:
                    # get the number of anomalies detected
                    df_output = pd.concat([df_output, df_anomalies], ignore_index=True)
                    log.info(
                        f"# of anomaly detected for app_id {app_id}: {len(df_anomalies)} for time {df_anomalies['AnomalyEventTime'].unique()}"
                    )
                else:
                    log.info(f"no anomaly detected for app_id {app_id}")

        except Exception as e:
            # Create a cloud watch alert for this error
            log.error(
                f"Error in run_inferencing for app_id={app_id}: {e}", exc_info=True
            )
            raise  # Re-raise to propagate error to Lambda
    # convert the anomaly dataframe to json and send to kinesis stream
    if len(df_output) > 0:
        try:
            send_records_to_kinesis(
                df_output, environ.get("KINESIS_STREAM_NAME"), region
            )
            log.info(f"Inferencing complete. Total anomalies sent from inferencing to Kinesis: {len(df_output)}")
        except Exception as e:
            # Create a cloud watch alert for this error
            log.error(f"Error sending records to Kinesis: {e}", exc_info=True)
            raise  # Re-raise to propagate error to Lambda
    else:
        log.info("no anomaly detected for all app_ids")

    return

# ============================================================================
# dip-ingestion-platform/mod-ml/aaa-inferencing-lambda/lib/aws_utils_lambda.py
# ============================================================================
import boto3
from botocore.exceptions import ClientError
import json
import psycopg2 as psql
import logging
import re

log = logging.getLogger(__name__)
log.propagate = True


def get_secrets(
    prefix: str,
    pod_name: str,
    db_name: str,
    region: str,
    session: boto3.session.Session,
) -> dict:
    db_type = {"aurora": "postgres", "redshift": "properties"}
    secret_name = f"{prefix}-{pod_name}-{db_type[db_name]}"
    sm_client = session.client(service_name="secretsmanager", region_name=region)
    secret_value_response = sm_client.get_secret_value(SecretId=secret_name)
    secrets = secret_value_response["SecretString"]
    secrets = json.loads(secrets)
    return secrets


def get_parameters(
    prefix: str,
    pod_name: str,
    db_name: str,
    region: str,
    session: boto3.session.Session,
) -> dict:
    db_type = {"aurora": "postgres", "redshift": "redshift"}
    parameter_name = f"{prefix}-{pod_name}-{db_type[db_name]}"
    ssm_client = session.client(service_name="ssm", region_name=region)
    parameter_value_response = ssm_client.get_parameter(Name=parameter_name)[
        "Parameter"
    ]["Value"]
    parameters = json.loads(parameter_value_response)
    return parameters


def get_aurora_postgresql_connection(prefix: str, pod_name: str, region: str):
    """
    Creates PostgreSQL Connector cursor, that can be used to send SQL queries.
    Login value are connected

    Args:
        prefix: the project Prefix, needed for the Aurora PostgreSQL credentials
        pod_name: the project PodName, needed for the Aurora PostgreSQL credentials
        region: region for Boto3 client

    Returns:
        PostgreSQL Connector cursor, that can be used to send SQL queries
    """
    try:
        session = boto3.session.Session()
        secrets = get_secrets(prefix, pod_name, "aurora", region, session)
        parameters = get_parameters(prefix, pod_name, "aurora", region, session)

        # reader_uri = parameters['devicemgmt.postgres.jdbc.url.reader']
        writer_uri = parameters["devicemgmt.postgres.jdbc.url.writer"]
        host, port, dbname = re.match(
            r"jdbc:postgresql://(.*?):(\d+)/(.*)", writer_uri
        ).groups()
        parameters = {"host": host, "port": port, "dbname": dbname}
        conn = psql.connect(
            host=parameters["host"],
            port=int(parameters["port"]),
            user=secrets["devicemgmt.postgres.jdbc.username"],
            password=secrets["devicemgmt.postgres.jdbc.password"],
            database=parameters["dbname"],
        )
    except ClientError as ce:
        log.error(
            "ERROR: Unexpected error: Could not fetch secrets "
            f"from secret manager: {ce}"
        )
        raise
    except Exception as exc:
        log.error(f"ERROR: Lambda initialization failed: {exc}")
        raise
    return conn


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key

# ============================================================================
# dip-ingestion-platform/mod-adapters/tripleagold/src/main/scala/com/vz/isp/adapters/triplea/gold/TripleaGoldJob.scala
# ============================================================================

import base64
import json
import logging

import boto3
from pandas import DataFrame
from splunk_hec_handler import SplunkHecHandler

from lib.app_config import AppConfig
from lib import process_anomaly


# This filter excludes logs from boto3 and botocore
class ExcludeBotoLogsFilter(logging.Filter):
    def filter(self, record):
        # Exclude logs from boto3 and botocore
        return not (
            record.name.startswith("boto3") or record.name.startswith("botocore")
        )


def setup_logging(config: AppConfig) -> logging.Logger:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    if config.splunk_host and config.splunk_token:
        try:
            splunk_handler = SplunkHecHandler(
                host=config.splunk_host,
                token=config.splunk_token,
                port=config.splunk_port,
                proto=config.splunk_proto,
                index=config.splunk_index or "main",
                source=config.splunk_source,
                sourcetype=config.splunk_sourcetype,
                ssl_verify=False,
                batch_size_count=20,
            )
            splunk_handler.addFilter(ExcludeBotoLogsFilter())
            splunk_handler.setLevel(logging.INFO)
            root_logger.addHandler(splunk_handler)
        except Exception as e:
            root_logger.error(
                f"Failed to initialize Splunk HEC handler: {e}", exc_info=True
            )
    return root_logger


# =================================================================================
# Global Initializations (Lambda Cold Start)
# =================================================================================
try:
    CONFIG = AppConfig.from_env()
    LOGGER = setup_logging(CONFIG)
    KINESIS_CLIENT = boto3.client("kinesis")
    SAGEMAKER_CLIENT = boto3.client("sagemaker")
    LOADED_MODELS = {}
except Exception as e:
    initialization_logger = logging.getLogger()
    initialization_logger.setLevel(logging.INFO)
    initialization_logger.error(f"CRITICAL: Failed during Lambda initialization: {e}")
    CONFIG = None


def ensure_splunk_handler(config: AppConfig, logger: logging.Logger):
    if (
        not any(isinstance(h, SplunkHecHandler) for h in logger.handlers)
        and config.splunk_host
        and config.splunk_token
    ):
        try:
            splunk_handler = SplunkHecHandler(
                host=config.splunk_host,
                token=config.splunk_token,
                port=config.splunk_port,
                proto=config.splunk_proto,
                index=config.splunk_index,
                source=config.splunk_source,
                sourcetype=config.splunk_sourcetype,
                ssl_verify=False,
                batch_size=20,
                flush_interval=0,
                queue_size=0,
                run_async=False,
            )
            splunk_handler.setLevel(logging.INFO)
            logger.addHandler(splunk_handler)
        except Exception as e:
            logger.error(
                f"Handler Failed to re-attach SplunkHecHandler: {e}", exc_info=True
            )


def lambda_handler(event, context):

    if CONFIG is None:
        LOGGER.error(
            "CRITICAL: Lambda initialization failed. "
            "Please check the environment variables and configuration."
        )
        # Return an error response to indicate that the Lambda function failed to initialize
        return {
            "statusCode": 500,
            "body": "Internal Server Error: Lambda initialization failed.",
        }
    ensure_splunk_handler(CONFIG, LOGGER)

    data = [
        json.loads(base64.b64decode(record["kinesis"]["data"]).decode("utf-8"))
        for record in event["Records"]
    ]
    df = DataFrame(data)
    if "app_id" not in df.columns:
        LOGGER.error("Input data is missing required 'app_id' column.")
        return {
            "statusCode": 400,
            "body": "Input data is missing required 'app_id' column.",
        }

    else:
        LOGGER.info(f"number of records read in: {len(df)}")
        try:
            process_anomaly.run_inferencing(loaded_models=LOADED_MODELS, df=df)
            return {
                "statusCode": 200,
                "body": "Successfully processed {} records.".format(df.shape[0]),
            }
        except Exception as e:
            LOGGER.error(
                f"Error in process_anomaly.run_inferencing: {e}", exc_info=True
            )
            return {
                "statusCode": 500,
                "body": f"Internal Server Error: {str(e)}",
            }

# =====================================================================================================================
# dip-ingestion-platform/mod-adapters/tripleagold/src/main/scala/com/vz/isp/adapters/triplea/gold/TripleaGoldJob.scala
# =====================================================================================================================
package com.vz.isp.adapters.triplea.gold

import com.amazonaws.services.glue.GlueContext
import com.amazonaws.services.glue.util.{GlueArgParser, Job}
import com.amazonaws.services.kinesis.{AmazonKinesis, AmazonKinesisClientBuilder}
import com.amazonaws.services.kinesis.model.{ProvisionedThroughputExceededException, PutRecordsRequest, PutRecordsRequestEntry}

import scala.collection.JavaConverters._
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.vz.isp.adapters.triplea.gold.KinesisUtils.writeToKinesis
import org.apache.commons.codec.digest.DigestUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, col, count, countDistinct, desc, expr, format_string, hour, lag, last, lit, max, mean, percentile_approx, pow, row_number, sqrt, stddev_pop, sum, to_date, to_timestamp, when}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}

import java.net.URI
import java.nio.ByteBuffer
import java.time.{LocalDateTime, ZoneOffset}
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit
import java.util.concurrent.TimeUnit
import scala.collection.JavaConverters.mapAsJavaMapConverter
import scala.util.{Failure, Random, Success, Try}

sealed trait ColumnDefinition {
  def name: String
}

object AAAEventColumns {
  case object AppId extends ColumnDefinition { val name = "app_id" }
  case object DeviceId extends ColumnDefinition { val name = "device_id" }
  case object EventDate extends ColumnDefinition { val name = "event_date" }
  case object OutputBytes extends ColumnDefinition { val name = "output_bytes" }
  case object InputBytes extends ColumnDefinition { val name = "input_bytes" }
  case object RecordType extends ColumnDefinition { val name = "record_type" }
  case object SessionTime extends ColumnDefinition { val name = "session_time" }
  case object SessionType extends ColumnDefinition { val name = "session_type" }
  case object SessionId extends ColumnDefinition { val name = "session_id" }
  case object Year extends ColumnDefinition { val name = "year" }
  case object Month extends ColumnDefinition { val name = "month" }
  case object Day extends ColumnDefinition { val name = "day" }
  case object Hour extends ColumnDefinition { val name = "hour" }
  case object Minute extends ColumnDefinition { val name = "minute" }
}

trait DataSource {
  @transient lazy val LOGGER: Logger = Logger.getLogger(getClass.getName)

  def load(): DataFrame
}

trait DataSink {
  @transient lazy val LOGGER: Logger = Logger.getLogger(getClass.getName)

  def write(dataFrame: DataFrame): Unit
}

class ParquetSource(sparkSession: SparkSession, path: String, filterExpression: Option[String]) extends DataSource {
  override def load(): DataFrame = {
    LOGGER.info(s"Loading dataframe from path=$path")
    val df = sparkSession.read.option("basePath", path).parquet(path)
    filterExpression match {
      case Some(expression) =>
        LOGGER.info(s"Loading dataframe with filter expr=$filterExpression")
        df.filter(expr(expression))
      case None => df
    }
  }
}

class GlueCatalogSource(glueContext: GlueContext, database: String, table: String) extends DataSource {
  override def load(): DataFrame = {
    glueContext.getCatalogSource(database, table).getDynamicFrame().toDF()
  }
}

class PartitionedParquetSink(path: String, partitionColumns: List[String], numPartitions: Int, orderingColumns: List[String] = List.empty) extends DataSink {
  override def write(dataFrame: DataFrame): Unit = {
    LOGGER.info(s"Writing dataframe to path=$path")
    LOGGER.info(s"Writing dataframe partitioned by partitionColumns=$partitionColumns, orderingColumns=$orderingColumns, numPartitions=$numPartitions")
    val df: DataFrame = dataFrame.repartition(numPartitions, partitionColumns.map(dataFrame.col): _*).sortWithinPartitions(orderingColumns.map(dataFrame.col): _*)

    if(!dataFrame.isEmpty){
      df.write
        .partitionBy(partitionColumns: _*)
        .mode("append")
        .parquet(path)
    } else {
      LOGGER.info("Dataframe is empty")
    }
  }
}

object KinesisUtils {
  @transient lazy val LOGGER: Logger = Logger.getLogger(getClass.getName)
  private val BASE_RETRY_DELAY_MS = 1000
  private val MAX_RETRIES = 5
  private val BATCH_SIZE = 25

  def writeToKinesis(df: DataFrame, streamName: String): Unit = {
    df.repartition(32).foreachPartition { partition: Iterator[Row] =>
      val kinesisClient: AmazonKinesis = AmazonKinesisClientBuilder.standard().build()
      val mapper = new ObjectMapper()
      mapper.registerModule(DefaultScalaModule)
      val partitionNo = TaskContext.getPartitionId()
      LOGGER.info(s"Processing partition number: $partitionNo")

      var batchNo = 0
      partition.grouped(BATCH_SIZE).foreach { batch =>
        val putRecordsRequest = new PutRecordsRequest()
        putRecordsRequest.setStreamName(streamName)

        val records = batch.map { row =>
          val record = row.getValuesMap(row.schema.fieldNames)
          val data = mapper.writeValueAsString(record)

          val appId = record.getOrElse(AAAEventColumns.AppId.name, 0).toString
          val deviceId = record.getOrElse(AAAEventColumns.DeviceId.name, 0).toString
          val salt = scala.util.Random.nextInt(10000)
          val rawKey = s"${appId}_${deviceId}_$salt"
          val partitionKey = DigestUtils.md5Hex(rawKey)

          val entry = new PutRecordsRequestEntry()
          entry.setData(ByteBuffer.wrap(data.getBytes()))
          entry.setPartitionKey(partitionKey)
          entry
        }.toList.asJava

        putRecordsRequest.setRecords(records)

        var attempt = 0
        var success = false
        while (attempt < MAX_RETRIES && !success) {
          Try {
            kinesisClient.putRecords(putRecordsRequest)
          } match {
            case Success(result) =>
              success = true
              val failedCount = result.getFailedRecordCount
              val usedShards = result.getRecords.asScala.map(_.getShardId).toSet
              LOGGER.info(s"Partition $partitionNo, Batch $batchNo: $failedCount failed records, used shards: ${usedShards.mkString(", ")}")
            case Failure(e: ProvisionedThroughputExceededException) =>
              attempt += 1
              val baseDelay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt).toLong
              val jitter = scala.util.Random.nextInt(200) + 50 // adds 50–250 ms
              val delay = baseDelay + jitter
              LOGGER.warn(s"ProvisionedThroughputExceededException: partition# $partitionNo, batch# $batchNo. Retrying in $delay ms (attempt $attempt/$MAX_RETRIES)")
              TimeUnit.MILLISECONDS.sleep(delay)
            case Failure(e) =>
              LOGGER.error(s"Failed to put records to Kinesis for partition# $partitionNo, batch# $batchNo", e)
              throw e
          }
        }

        if (!success) {
          throw new RuntimeException(s"Failed to put record to Kinesis after $MAX_RETRIES attempts for partition# $partitionNo, batch# $batchNo")
        }

        TimeUnit.MILLISECONDS.sleep(200)
        batchNo += 1
      }
    }
  }
}

class KinesisSink(streamName: String) extends DataSink {

  override def write(dataFrame: DataFrame): Unit = {
    dataFrame.foreachPartition((rows: Iterator[Row]) => ???)
  }
}

object TripleaGoldJob {
  @transient private lazy val LOGGER: Logger = Logger.getLogger(getClass.getName)

  def main(sysArgs: Array[String]): Unit = {
    LOGGER.info("TripleaGoldJob started")
    val spark: SparkSession = getSparkSession
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
    val glueContext: GlueContext = new GlueContext(spark.sparkContext)
    val glueOptions = collectGlueOptions(sysArgs)
    val args = GlueArgParser.getResolvedOptions(sysArgs, glueOptions.toArray)

    Job.init(args("JOB_NAME"), glueContext, args.asJava)

    val s3InputPath = s"s3://${args("INPUT_BUCKET")}/"
    val s3FeaturesOutputPath = s"s3://${args("OUTPUT_BUCKET")}/${args("DATASET_AGGREGATED")}/"
    val s3HistoricStatsPath = s"s3://${args("OUTPUT_BUCKET")}/${args("DATASET_HISTORIC")}/"
    val datetimeColumn = args("DATETIME_COLUMN")
    val endTimeStamp = args.get("CURRENT_TIME")
    val targetStreamName = args("TARGET_KINESIS")
    val appIds = None
    val jobTimeStamp: LocalDateTime = getJobTimeStamp(endTimeStamp)

    LOGGER.info(f"Job Parameters")
    LOGGER.info(f"Job Parameters - Job run timestamp: ${jobTimeStamp.toString}")
    LOGGER.info(f"Job Parameters - Event data location: $s3InputPath")
    LOGGER.info(f"Job Parameters - Feature data location: $s3FeaturesOutputPath")
    LOGGER.info(f"Job Parameters - Kinesis output: $targetStreamName")
    LOGGER.info(f"Job Parameters - Historic stats data location: $s3HistoricStatsPath")
    LOGGER.info(f"Job Parameters - Datetime column used for event processing: $datetimeColumn")

    //  Previous hour is processed: load raw events from current and previous hours partition to handle late events
    val eventDf: DataFrame = new ParquetSource(spark, s3InputPath, createDateTimeAndIdFilter(datetimeColumn, jobTimeStamp, appIds))
      .load()
      .withColumn(AAAEventColumns.AppId.name, col(AAAEventColumns.AppId.name).cast(IntegerType))
      .withColumn(AAAEventColumns.DeviceId.name, col(AAAEventColumns.DeviceId.name).cast(IntegerType))
    LOGGER.info(f"${eventDf.count()} events being processed")
    //  Find latest available partition before the run's hour in persisted stats from previous runs
    val prevStats: DataFrame = getPreviousStats(spark, s3HistoricStatsPath, jobTimeStamp)
      .withColumn(AAAEventColumns.AppId.name, col(AAAEventColumns.AppId.name).cast(IntegerType))
      .withColumn(AAAEventColumns.DeviceId.name, col(AAAEventColumns.DeviceId.name).cast(IntegerType))
      .transform(repartitionByDevice("date"))
    //    Take last 7 aggregation results for rolling avg calculation
    prevStats.cache()
    //    Take last 7 aggregation results for rolling avg calculation
    val previousNAggregations: DataFrame = getLastNAggregations(spark, s3FeaturesOutputPath, 7, jobTimeStamp)
      .withColumn(AAAEventColumns.AppId.name, col(AAAEventColumns.AppId.name).cast(IntegerType))
      .withColumn(AAAEventColumns.DeviceId.name, col(AAAEventColumns.DeviceId.name).cast(IntegerType))
    val (preprocessedDf, newPrevDf) = processData(eventDf, prevStats, previousNAggregations, datetimeColumn, jobTimeStamp)
    val parquetSinkPreprocessed = new PartitionedParquetSink(s3FeaturesOutputPath, List(AAAEventColumns.AppId.name, "date", "hour"), 10)
    parquetSinkPreprocessed.write(preprocessedDf)
    writeToKinesis(preprocessedDf, targetStreamName)
    //    new KinesisSink(targetStreamName).write(preprocessedDf)
    val parquetSinkHistoric = new PartitionedParquetSink(s3HistoricStatsPath, List("date", "hour"), 10)
    //  Construct dataframe with upsert like operation to update historical snapshot for stats
    parquetSinkHistoric.write(newPrevDf.transform(upsertIntoHistoricDf(prevStats, jobTimeStamp)))
  }

  def collectGlueOptions(sysArgs: Array[String]): Seq[String] = {
    var glueOptions: Seq[String] = Seq("JOB_NAME", "INPUT_BUCKET", "OUTPUT_BUCKET", "DATASET_AGGREGATED", "DATASET_HISTORIC", "TARGET_KINESIS", "DATETIME_COLUMN")
    val optionalParams: Seq[String] = Seq("CURRENT_TIME"
      , "APP_IDS"
    )
    optionalParams.foreach(optionalParam => {
      if (sysArgs.exists(_.startsWith(s"--$optionalParam"))) {
        glueOptions = glueOptions :+ optionalParam
      }
    })
    glueOptions
  }

  def checkPathExists(sparkSession: SparkSession, path: String): Boolean = {
    val fs = FileSystem.get(new URI(path), sparkSession.sparkContext.hadoopConfiguration)
    val basePath = new Path(path)
    fs.exists(basePath)
  }

  def getLastNAggregations(spark: SparkSession, path: String, noOfRows: Int, timeStamp: LocalDateTime): DataFrame = {
    if (checkPathExists(spark, path)) {
      val lastNHoursFilter = (1 to noOfRows).map(lag => timeStamp.minusHours(lag)).toList.map(t => f"(date='${t.toLocalDate.toString}' AND hour='${t.getHour}%02d')").mkString(" OR ")
      new ParquetSource(spark, path, Some(lastNHoursFilter))
        .load()
        .select(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour", "max_zero_bytes_Count")
      //        .withColumnRenamed("zeroBytesSession", "max_zero_bytes_Count")
    } else {
      LOGGER.info(f"Dataset at $path is not initialized")
      val schema = StructType(Array(
        StructField(AAAEventColumns.AppId.name, IntegerType, nullable = false),
        StructField(AAAEventColumns.DeviceId.name, IntegerType, nullable = false),
        StructField("date", StringType, nullable = false),
        StructField("hour", StringType, nullable = false),
        StructField("max_zero_bytes_Count", IntegerType, nullable = true)
      ))
      spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
    }
  }

  def getPreviousStats(spark: SparkSession, path: String, endTimeStamp: LocalDateTime): DataFrame = {
    getLatestAvailablePartitionOfHistoric(spark, path, endTimeStamp) match {
      case Some(value) =>
        val histSource = new ParquetSource(spark, path, Some(f"date='${value._1}' AND hour='${value._2}'"))
        histSource.load()
          .distinct()
          .withColumn("row_number", row_number().over(Window.partitionBy("app_id", "device_id").orderBy(desc("prev_count_session_count"))))
          .filter(col("row_number") === 1)
          .drop("row_number")
      case None =>
        LOGGER.info(f"Dataset at $path is not initialized")
        val schema = StructType(Array(
          StructField(AAAEventColumns.AppId.name, IntegerType, nullable = false),
          StructField(AAAEventColumns.DeviceId.name, IntegerType, nullable = false),
          StructField("date", StringType, nullable = false),
          StructField("hour", StringType, nullable = false),
          StructField("prev_mean_session_count", FloatType, nullable = true),
          StructField("prev_std_session_count", FloatType, nullable = true),
          StructField("prev_count_session_count", IntegerType, nullable = true),
          StructField("prev_mean_max_shortLength", FloatType, nullable = true),
          StructField("prev_std_max_shortLength", FloatType, nullable = true),
          StructField("prev_count_max_shortLength", IntegerType, nullable = true),
          StructField(f"prev_mean_${AAAEventColumns.SessionTime.name}", FloatType, nullable = true),
          StructField(f"prev_std_${AAAEventColumns.SessionTime.name}", FloatType, nullable = true),
          StructField(f"prev_count_${AAAEventColumns.SessionTime.name}", IntegerType, nullable = true)
        ))
        spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
    }
  }

  def upsertIntoHistoricDf(oldDf: DataFrame, timeStamp: LocalDateTime)(newDf: DataFrame): DataFrame = {
    oldDf
      .join(newDf, Seq(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name), "leftanti")
      .unionByName(newDf)
      .select(oldDf.columns.map( name => {
        if (name.contains("prev_mean") | name.contains("prev_std")) col(name).cast(FloatType)
        else if (name.contains("prev_count")) col(name).cast(IntegerType)
        else col(name)
      }): _*)
      .withColumn("date", lit(timeStamp.toLocalDate.toString))
      .withColumn("hour", lit(f"${timeStamp.getHour}%02d"))
      .withColumn(AAAEventColumns.AppId.name, col(AAAEventColumns.AppId.name).cast(IntegerType))
      .withColumn(AAAEventColumns.DeviceId.name, col(AAAEventColumns.DeviceId.name).cast(IntegerType))
      .toDF()
  }

  def getJobTimeStamp(timeStampStringOption: Option[String]): LocalDateTime = {
    timeStampStringOption match {
      case Some(value) => LocalDateTime.parse(value, DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
      case None => LocalDateTime.now(ZoneOffset.UTC)
    }
  }

  def getLatestAvailablePartitionOfHistoric(sparkSession: SparkSession, path: String, timeStamp: LocalDateTime): Option[(String, String)] = {
    val fs = FileSystem.get(new URI(path), sparkSession.sparkContext.hadoopConfiguration)
    val basePath = new Path(path)
    if(fs.exists(basePath)) {
      val partitions = fs
        .listStatus(basePath)
        .filter(_.isDirectory)
        .flatMap(status => {
          val date = status.getPath.getName.split("=")(1)
          val hourPartitionsPath = new Path(status.getPath.toString)
          fs.listStatus(hourPartitionsPath).filter(_.isDirectory).map(hourStatus => {
            val hour = hourStatus.getPath.getName.split("=")(1)
            (date, hour)
          })
        }).toSeq

      val filteredPartitions = partitions.filter {
        case (date, hour) => (date < timeStamp.toLocalDate.toString) || (date == timeStamp.toLocalDate.toString && hour < f"${timeStamp.getHour}%02d")
      }

      filteredPartitions.sortWith {
        case ((date, hour), (dateOther, hourOther)) => (date < dateOther) || (date == dateOther && hour < hourOther)
      }.lastOption
    } else {
      None
    }
  }

  def generateTimeSequence(start: LocalDateTime, end: LocalDateTime): Stream[String] = {
    if (start.isAfter(end)) Stream.empty
    else {
      val current = f"(year=${start.getYear} AND month=${start.getMonthValue}%02d AND day=${start.getDayOfMonth}%02d AND hour=${start.getHour}%02d)"
      current #:: generateTimeSequence(start.plusHours(1), end)
    }
  }

  def createDateTimeAndIdFilter(datetimeColumn: String, endTimeStamp: LocalDateTime, appIds: Option[String]): Option[String] = {
    val startTimeStamp = endTimeStamp.minus(1, ChronoUnit.HOURS)
    val partFilterSequence = generateTimeSequence(startTimeStamp, endTimeStamp).mkString("(", " OR ", ")")
    val eventTimeFilter = s" AND ($datetimeColumn >= '${endTimeStamp.minusHours(1).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:00:00"))}' AND $datetimeColumn < '${endTimeStamp.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:00:00"))}')"
    val dateTimeFilter = partFilterSequence + eventTimeFilter
    Some(appIds.map(ids => s"${AAAEventColumns.AppId.name} IN ($ids) AND $dateTimeFilter").getOrElse(dateTimeFilter))
  }

  private def getSparkSession: SparkSession = {
    SparkSession.builder()
      .appName("TripleaGoldJob Batch Job")
      .getOrCreate()
  }

  def processData(dataframe: DataFrame, prevStatsDf: DataFrame, previousNAggregations: DataFrame, dateTimeColumn: String, jobTimeStamp: LocalDateTime): (DataFrame, DataFrame) = {
    LOGGER.info(f"${dataframe.count()} events being processed")
    val preprocessedEvents = dataframe
      .transform(processUTCDate(dateTimeColumn))
      .transform(deduplicateByColumns(List(AAAEventColumns.AppId.name,
        AAAEventColumns.DeviceId.name,
        dateTimeColumn,
        AAAEventColumns.RecordType.name,
        AAAEventColumns.SessionId.name)))
      .transform(createUsage)
      .transform(repartitionByDevice(dateTimeColumn))

    val frequentSessionFlagDf = preprocessedEvents.transform(withFrequentSessionFlag(prevStatsDf))
    val shortSessionFlagDf = preprocessedEvents.transform(withShortSessionFlag(prevStatsDf))
    val sessionTimeAnomalyFlagDf = preprocessedEvents.transform(withSessionTimeAnomalyFlag(prevStatsDf))
    val zeroSessionAnomalyFlagDf = preprocessedEvents.transform(withZeroByteSessionAnomalyFlag(previousNAggregations, jobTimeStamp))

    val joinColumns = Seq(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")

    val mergedDf = frequentSessionFlagDf
      .join(shortSessionFlagDf, joinColumns, "outer")
      .join(sessionTimeAnomalyFlagDf, joinColumns, "outer")
      .join(zeroSessionAnomalyFlagDf, joinColumns, "outer")
      .withColumn("hour", format_string("%02d", col("hour").cast(IntegerType)))

    val currentStatColumns = joinColumns ++ Seq("session_count", "expanding_mean_session_count", "AnomalySessionCount", "max_shortLength", "expanding_mean_max_shortLength", "shortSessionAnomaly", "AvgSessionLengthTime", AAAEventColumns.SessionTime.name, f"expanding_mean_${AAAEventColumns.SessionTime.name}", "sessionTimeAnomalyBasedOnHistory", "max_zero_bytes_Count", "rolling_mean_zero_bytes_Count", "ZerobytesUsagenAnomaly")
    val prevStatColumns = joinColumns ++ Seq("prev_mean_session_count", "prev_std_session_count", "prev_count_session_count", "prev_mean_max_shortLength", "prev_std_max_shortLength", "prev_count_max_shortLength", f"prev_mean_${AAAEventColumns.SessionTime.name}", f"prev_std_${AAAEventColumns.SessionTime.name}", s"prev_count_${AAAEventColumns.SessionTime.name}")

    val finalDf = mergedDf
      .select(currentStatColumns.map(col): _*)
      //      .withColumnRenamed("session_count", "NormalSessionCount")
      //      .withColumnRenamed("max_shortLength", "ShortSession")
      //      .withColumnRenamed("max_zero_bytes_Count", "zeroBytesSession")
      //      .withColumnRenamed("sessionTimeAnomalyBasedOnHistory", "sessionTimeAnomaly")
      .withColumn(AAAEventColumns.AppId.name, col(AAAEventColumns.AppId.name).cast(IntegerType))
      .withColumn(AAAEventColumns.DeviceId.name, col(AAAEventColumns.DeviceId.name).cast(IntegerType))
      .withColumn("date", col("date").cast(StringType))
      .withColumn("hour", col("hour").cast(StringType))
      .withColumn("session_count", col("session_count").cast(LongType))
      .withColumn("expanding_mean_session_count", col("expanding_mean_session_count").cast(DoubleType))
      .withColumn("AnomalySessionCount", col("AnomalySessionCount").cast(IntegerType))
      .withColumn("max_shortLength", col("max_shortLength").cast(LongType))
      .withColumn("expanding_mean_max_shortLength", col("expanding_mean_max_shortLength").cast(DoubleType))
      .withColumn("shortSessionAnomaly", col("shortSessionAnomaly").cast(IntegerType))
      .withColumn("AvgSessionLengthTime", col("AvgSessionLengthTime").cast(DoubleType))
      .withColumn("session_time", col("session_time").cast(DoubleType))
      .withColumn("expanding_mean_session_time", col("expanding_mean_session_time").cast(DoubleType))
      .withColumn("sessionTimeAnomalyBasedOnHistory", col("sessionTimeAnomalyBasedOnHistory").cast(IntegerType))
      .withColumn("max_zero_bytes_Count", col("max_zero_bytes_Count").cast(LongType))
      .withColumn("rolling_mean_zero_bytes_Count", col("rolling_mean_zero_bytes_Count").cast(DoubleType))
      .withColumn("ZerobytesUsagenAnomaly", col("ZerobytesUsagenAnomaly").cast(IntegerType))

    val newPrevDf = mergedDf.select(prevStatColumns.map(col): _*)

    (finalDf, newPrevDf)
  }

  private def processUTCDate(dateTimeColumn: String)(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn(dateTimeColumn, to_timestamp(col(dateTimeColumn)))
      .withColumn("hour", hour(col(dateTimeColumn)))
      .withColumn("date", to_date(col(dateTimeColumn)))
  }

  private def deduplicateByColumns(columnList: List[String])(dataFrame: DataFrame): DataFrame = {
    val result = dataFrame.dropDuplicates(columnList)
    LOGGER.info(s"Number of events after deduplication: ${result.count()}")
    result
  }

  private def createUsage(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("Bytes_usage", col(AAAEventColumns.InputBytes.name) + col(AAAEventColumns.OutputBytes.name))
      .drop(AAAEventColumns.InputBytes.name, AAAEventColumns.OutputBytes.name)
  }

  private def repartitionByDevice(dateTimeColumn: String)(dataFrame: DataFrame): DataFrame = {
    val df = dataFrame
      .repartition(col(AAAEventColumns.AppId.name), col(AAAEventColumns.DeviceId.name))
      .orderBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, dateTimeColumn)

    dataFrame.cache()
    df
  }

  def withExpandingStatistics(prevStatsDf: DataFrame, targetColumn: String)(dataFrame: DataFrame): DataFrame = {
    val window = Window.partitionBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name).orderBy("date", "hour")
    val prevMeanColumn = s"prev_mean_${targetColumn}"
    val prevStdColumn = s"prev_std_${targetColumn}"
    val prevCountColumn = s"prev_count_${targetColumn}"

    dataFrame
      .join(prevStatsDf.select(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, prevMeanColumn, prevStdColumn, prevCountColumn), Seq(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name), "left_outer")
      .na.fill(Map(prevMeanColumn -> 0.0, prevStdColumn -> 0.0, prevCountColumn -> 0))
      .withColumn("count", col(prevCountColumn) + count(col(targetColumn)).over(window))

      //    new_mean = (prev_mean * prev_count + new_value) / (prev_count + 1)
      .withColumn("expanding_mean_" + targetColumn, col(prevMeanColumn).cast(DoubleType))
      .withColumn(prevMeanColumn, (col(prevMeanColumn) * col(prevCountColumn) + col(targetColumn)) / col("count"))
      //    new_variance = (prev_std ** 2 * prev_count + (new_value - prev_mean) * (new_value - new_mean)) / (prev_count + 1)
      //    new_std = np.sqrt(new_variance)
      .withColumn("expanding_std_" + targetColumn, col(prevStdColumn).cast(DoubleType))
      .withColumn(prevStdColumn,
        sqrt((pow(col(prevStdColumn), 2) * col(prevCountColumn) + (col(targetColumn) - col("expanding_mean_" + targetColumn)) * (col(targetColumn) - col(prevMeanColumn))) / col("count"))
      )
      .drop(prevCountColumn)
      .withColumnRenamed("count", prevCountColumn)
  }

  private def withFrequentSessionFlag(prevStatsDf: DataFrame)(dataFrame: DataFrame): DataFrame = {
    //    val window = Window.partitionBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name).orderBy("date", "hour").rowsBetween(Window.unboundedPreceding, -1)
    val absDiffThreshold = 4
    val zScoreThreshold = 3
    val targetColumn = "session_count"
    val meanColumn = s"expanding_mean_$targetColumn"
    val stdColumn = s"expanding_std_$targetColumn"

    dataFrame.groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(countDistinct(AAAEventColumns.SessionId.name).alias(targetColumn))
      //      .withColumn("expanding_mean_session_cnt", mean("session_count").over(window))
      //      .withColumn("expanding_std", stddev_pop("session_count").over(window))
      //      .withColumn("expanding_std", when(col("expanding_std") === 0, lit(null)).otherwise(col("expanding_std")))
      .transform(withExpandingStatistics(prevStatsDf, targetColumn))
      .withColumn(stdColumn, when(col(stdColumn) === 0, lit(null)).otherwise(col(stdColumn)))
      .withColumn("z_score_session_cnt", (col(targetColumn) - col(meanColumn)) / col(stdColumn))
      .withColumn("AnomalySessionCount", when((functions.abs(col("z_score_session_cnt")) > zScoreThreshold) && (functions.abs((col(targetColumn) - col(meanColumn))) > absDiffThreshold), 1).otherwise(0))
      .na.fill(Map("z_score_session_cnt" -> 0, "AnomalySessionCount" -> 0))
      .drop("z_score_session_cnt", stdColumn)
  }

  private def withShortSessionFlag(prevStatsDf: DataFrame)(dataFrame: DataFrame): DataFrame = {
    val absDiffThreshold = 4
    val zScoreThreshold = 3
    val targetColumn = "max_shortLength"
    val meanColumn = s"expanding_mean_$targetColumn"
    val stdColumn = s"expanding_std_$targetColumn"

    val dfWithFlag = dataFrame
      .withColumn("shortsessionLength", when(col(AAAEventColumns.SessionTime.name).cast("int") <= 60, 1).otherwise(0))

    val avgSessionTimeDf = dfWithFlag
      .groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(avg(when(col("shortsessionLength") === 1, col(AAAEventColumns.SessionTime.name)).otherwise(null)).alias("AvgSessionLengthTime"))

    dfWithFlag
      .groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(sum("shortsessionLength").alias("shortSessionCount"))
      .groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(max("shortSessionCount").alias(targetColumn))
      .transform(withExpandingStatistics(prevStatsDf, targetColumn))
      .withColumn(stdColumn, when(col(stdColumn) === 0, lit(null)).otherwise(col(stdColumn)))
      .withColumn("z_score_short_session", (col(targetColumn) - col(meanColumn)) / col(stdColumn))
      .withColumn("shortSessionAnomaly", when((functions.abs(col("z_score_short_session")) > zScoreThreshold) && (functions.abs((col(targetColumn) - col(meanColumn))) > absDiffThreshold), 1).otherwise(0))
      .na.fill(Map("z_score_short_session" -> 0, "shortSessionAnomaly" -> 0))
      .drop("z_score_short_session", stdColumn)
      .join(avgSessionTimeDf, Seq(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour"), "left_outer")
  }

  private def withSessionTimeAnomalyFlag(prevStatsDf: DataFrame)(dataFrame: DataFrame): DataFrame = {
    val upperZScoreThreshold = 3
    val lowerZScoreThreshold = -1
    val targetColumn = AAAEventColumns.SessionTime.name
    val meanColumn = s"expanding_mean_$targetColumn"
    val stdColumn = s"expanding_std_$targetColumn"

    val aggDf = dataFrame
      .groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(mean(AAAEventColumns.SessionTime.name).alias(targetColumn))
      .transform(withExpandingStatistics(prevStatsDf, targetColumn))
      .withColumn(stdColumn, when(col(stdColumn) === 0, lit(null)).otherwise(col(stdColumn)))
      .withColumn("z_score_session_Time", (col(targetColumn) - col(meanColumn)) / col(stdColumn))
      .withColumn("diff", col(targetColumn) - col(meanColumn))

    val percentiles = aggDf.agg(percentile_approx(col("diff"), lit(0.95), lit(ApproximatePercentile.DEFAULT_PERCENTILE_ACCURACY)).alias("95th"), percentile_approx(col("diff"), lit(0.1), lit(ApproximatePercentile.DEFAULT_PERCENTILE_ACCURACY)).alias("10th")).collect()
    val upperAbsDiffThreshold = percentiles(0).getAs[Double]("95th")
    val lowerAbsDiffThreshold = percentiles(0).getAs[Double]("10th")

    aggDf
      .withColumn("sessionTimeAnomalyBasedOnHistory", when(((col("z_score_session_Time") > upperZScoreThreshold) && (col(targetColumn) - col(meanColumn) > upperAbsDiffThreshold)) || ((col("z_score_session_Time") < lowerZScoreThreshold) && (col(targetColumn) - col(meanColumn) < lowerAbsDiffThreshold)), 1).otherwise(0))
      .na.fill(Map("z_score_session_Time" -> 0, "sessionTimeAnomalyBasedOnHistory" -> 0))
      .drop("z_score_session_Time", stdColumn, "diff")
  }

  private def withZeroByteSessionAnomalyFlag(previousNAggregations: DataFrame, jobTimeStamp: LocalDateTime)(dataFrame: DataFrame): DataFrame = {
    val absDiffThreshold = 5
    val zScoreThreshold = 1
    val tolerance = 1e5
    val jumpThreshold = 5
    val zeroByteSessionCountColumn = "max_zero_bytes_Count"

    val windowRolling = Window.partitionBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name).orderBy("date", "hour").rowsBetween(-7, 0)
    val windowHourly = Window.partitionBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name).orderBy("date", "hour")

    val prevFilteredDf = previousNAggregations
      .join(dataFrame.select(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name).distinct(), Seq(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name))
      .select(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour", zeroByteSessionCountColumn)
    dataFrame
      .withColumn("zero_bytes_flag", when((col("Bytes_usage") === 0), 1).otherwise(0))
      .groupBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date", "hour")
      .agg(sum("zero_bytes_flag").alias(zeroByteSessionCountColumn))
      //      .withColumn("max_zero_bytes_Count", max("zero_bytes_flagCount").over(Window.partitionBy(AAAEventColumns.AppId.name, AAAEventColumns.DeviceId.name, "date")))
      .withColumn(zeroByteSessionCountColumn, when(col(zeroByteSessionCountColumn) === 0, lit(null)).otherwise(col(zeroByteSessionCountColumn)))
      .unionByName(prevFilteredDf)
      .withColumn("rolling_mean", avg(zeroByteSessionCountColumn).over(windowRolling))
      .withColumn("rolling_std", stddev_pop(zeroByteSessionCountColumn).over(windowRolling))
      .na.fill(Map(zeroByteSessionCountColumn -> 0, "rolling_mean" -> 0, "rolling_std" -> 0))
      .withColumn("z_score_bytes_usage", (col(zeroByteSessionCountColumn) - col("rolling_mean")) / col("rolling_std"))
      .na.fill(Map("z_score_bytes_usage" -> 0, zeroByteSessionCountColumn -> 0))
      .withColumn(zeroByteSessionCountColumn, when(col(zeroByteSessionCountColumn) === 0, lit(null)).otherwise(col(zeroByteSessionCountColumn)))
      .withColumn("prev_value", lag(zeroByteSessionCountColumn, 1).over(windowHourly))
      .withColumn("previous_non_zero", last("prev_value", ignoreNulls = true).over(windowHourly))
      .na.fill(Map(zeroByteSessionCountColumn -> 0))
      .withColumn("diff", col(zeroByteSessionCountColumn) - col("rolling_mean"))
      .withColumn("is_significant_jump", col(zeroByteSessionCountColumn) > (col("previous_non_zero") + jumpThreshold))
      .withColumn("ZerobytesUsagenAnomaly", when((col(zeroByteSessionCountColumn) > 0) && col(zeroByteSessionCountColumn) > absDiffThreshold, 1).otherwise(0))
      .withColumnRenamed("rolling_mean", "rolling_mean_zero_bytes_Count")
      .drop("rolling_std", "z_score_bytes_usage", "prev_value", "previous_non_zero", "is_significant_jump", "diff")
      .filter(col("date") === jobTimeStamp.minusHours(1).toLocalDate.toString && col("hour") === jobTimeStamp.minusHours(1).getHour)
  }
}

#==========================================
model_zoo
#==========================================
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ---------------------------------------------------------------------
# PARAM VALIDATION (CATCH TYPOS EARLY)
# ---------------------------------------------------------------------
ALGO_PARAM_WHITELIST = {
    "isolation_forest": {
        "n_estimators", "max_samples", "max_features",
        "contamination", "bootstrap", "n_jobs", "random_state",
        "warm_start"
    },
    "lof": {
        "n_neighbors", "algorithm", "leaf_size", "metric", "p",
        "novelty", "n_jobs"
    },
    "autoencoder": {
        # placeholder keys – your external AE class will define usage
        "epochs", "batch_size", "lr", "weight_decay",
        "hidden_dims", "bottleneck", "dropout",
        "device", "seed", "early_stop_patience"
    },
}


def _validate_params(algo: str, params: Dict) -> None:
    a = algo.lower()
    if a not in ALGO_PARAM_WHITELIST:
        raise ValueError(f"Unsupported algorithm: {algo}")
    unknown = set(params) - ALGO_PARAM_WHITELIST[a]
    if unknown:
        raise ValueError(f"Unknown {algo} params: {sorted(unknown)}")


def _as_1d_float(x) -> np.ndarray:
    arr = np.asarray(x).reshape(-1).astype(float)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, copy=False)
    return arr


# ---------------------------------------------------------------------
# FACTORY: BUILD PIPELINE
# ---------------------------------------------------------------------
def build_pipeline(algo: str, params: Dict) -> Tuple[Pipeline, bool]:
    """
    Returns (pipeline, needs_fit_predict)
    - `needs_fit_predict=True` means model must use fit_predict (LOF classic mode).
    """
    a = algo.lower()
    _validate_params(a, params)

    # ----- Isolation Forest -----
    if a == "isolation_forest":
        model = IsolationForest(**params)
        pipe = Pipeline([("model", model)])  # No scaler – uses raw features
        return pipe, False

    # ----- LOF -----
    if a == "lof":
        novelty = bool(params.get("novelty", True))
        model = LocalOutlierFactor(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        # LOF in classic mode: needs fit_predict
        return pipe, (not novelty)

    # ----- Autoencoder (placeholder) -----
    if a == "autoencoder":
        # placeholder — will import actual AE implementation later
        try:
            from .autoencoder_model import AutoEncoderWrapper
        except ImportError:
            raise ImportError(
                "Autoencoder module not found. Please ensure 'autoencoder_model.py' "
                "defines AutoEncoderWrapper class with fit() and reconstruct()."
            )

        model = AutoEncoderWrapper(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        return pipe, False

    raise ValueError(f"Unknown algo: {algo}")


# ---------------------------------------------------------------------
# SCORING ADAPTER
# ---------------------------------------------------------------------
def anomaly_scores(algo: str, pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Return 1D float array where HIGHER = MORE ANOMALOUS.

    - IsolationForest:    -score_samples(X)
    - LOF (novelty=True): -decision_function(X)
    - LOF (classic):      -negative_outlier_factor_
    - Autoencoder:        handled by imported wrapper (reconstruction error)
    """
    a = algo.lower()

    # ----- Isolation Forest -----
    if a == "isolation_forest":
        raw = pipe["model"].score_samples(X)
        return _as_1d_float(-raw)

    # ----- LOF -----
    if a == "lof":
        model: LocalOutlierFactor = pipe["model"]
        if getattr(model, "novelty", False):
            raw = model.decision_function(X)  # higher = more normal
            return _as_1d_float(-raw)
        else:
            if not hasattr(model, "negative_outlier_factor_"):
                raise RuntimeError("LOF classic mode requires fit_predict(X) before scoring.")
            raw = model.negative_outlier_factor_
            return _as_1d_float(-raw)

    # ----- Autoencoder (placeholder) -----
    if a == "autoencoder":
        model = pipe["model"]
        if not hasattr(model, "reconstruct"):
            raise AttributeError("AutoEncoderWrapper must implement reconstruct(X).")
        recon = model.reconstruct(X)
        err = np.mean((X - recon) ** 2, axis=1)
        return _as_1d_float(err)

    raise ValueError(f"Unknown algo: {algo}")
    
#==========================================
run_experiment
#==========================================
# src/aaa/exp/run_experiment.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import mlflow
import yaml

# Local imports
from .model_zoo import build_pipeline, anomaly_scores


# ---------------------------
# Utilities
# ---------------------------
def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sha1_of_text(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:12]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_parquet_glob(glob_path: str | Path) -> pd.DataFrame:
    """Simple, dependency-free parquet loader over a glob."""
    import glob
    files = sorted(glob.glob(str(glob_path)))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {glob_path}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame, feature_yaml: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a feature spec YAML, produce:
      - Xdf: numeric feature frame used by the model
      - iddf: identifier columns for reporting (e.g., device_id, date, hour)
    Expected YAML format (example):
      features:
        id: ["device_id", "date", "hour"]
        input: ["session_count", "expanding_mean_session_count", ...]
    """
    feats = feature_yaml.get("features", {})
    id_cols = feats.get("id", [])
    input_cols = feats.get("input", [])

    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in data: {missing}")

    iddf = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    Xdf = df[input_cols].copy()

    # Basic sanitation: cast to float and fill NaNs
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.mean()).fillna(0.0)
    return Xdf, iddf


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    return {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_var": float(np.var(scores)),
    }


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), scores.shape[0]))
    return np.argpartition(scores, -k)[-k:][np.argsort(scores[np.argpartition(scores, -k)[-k:]])[::-1]]


# ---------------------------
# Main single-run executor
# ---------------------------
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser("AAA single experiment runner")
    ap.add_argument("--data", required=True, help="Parquet glob path, e.g. data_stream/processed/date=*/part.parquet")
    ap.add_argument("--features", required=True, help="Feature spec YAML")
    ap.add_argument("--config", required=True, help="Experiment YAML (algo + params)")
    ap.add_argument("--experiment-name", default="AAA-Experiments", help="MLflow experiment name")
    ap.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (optional)")
    ap.add_argument("--topk", type=int, default=200, help="Top-K anomalies to export as artifact")
    ap.add_argument("--register-model-name", default=None, help="Optional MLflow Model Registry name")
    args = ap.parse_args(argv)

    # MLflow init
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load files
    df = load_parquet_glob(args.data)
    feat_spec = read_yaml(args.features)
    cfg = read_yaml(args.config)

    algo: str = cfg.get("algo")
    params: Dict[str, Any] = cfg.get("params", {})
    metric_name: str = cfg.get("metric", "score_mean")   # purely informational for now
    maximize: bool = bool(cfg.get("maximize", True))
    feature_id = feat_spec.get("id", cfg.get("id", "fs_v1"))

    # Derive config hash for reproducibility tag
    cfg_text = yaml.safe_dump(cfg, sort_keys=True)
    cfg_hash = sha1_of_text(cfg_text)

    # Build features
    Xdf, iddf = build_feature_matrix(df, feat_spec)
    X = Xdf.to_numpy(dtype=float)

    # Build pipeline from model_zoo
    pipe, needs_fit_predict = build_pipeline(algo, params)

    # Start MLflow run
    run_name = f"{algo}:{cfg_hash}"
    with mlflow.start_run(run_name=run_name):
        # ---- Log params & tags
        mlflow.set_tag("algo", algo)
        mlflow.set_tag("feature_id", feature_id)
        mlflow.set_tag("config_hash", cfg_hash)
        mlflow.log_params(params)

        # ---- Fit
        t0 = time.time()
        if needs_fit_predict:
            # e.g., LOF (classic mode)
            _ = pipe.fit_predict(X)
        else:
            pipe.fit(X)
        fit_s = time.time() - t0
        mlflow.log_metric("fit_time_s", fit_s)

        # ---- Score (higher = more anomalous, per model_zoo contract)
        scores = anomaly_scores(algo, pipe, X)
        stats = summarize_scores(scores)
        mlflow.log_metrics(stats)

        # ---- Export artifacts (config, features, top-K)
        artifacts_dir = ensure_dir("artifacts/experiment")
        # Save exact experiment/config used
        cfg_out = artifacts_dir / f"config_{cfg_hash}.yaml"
        with open(cfg_out, "w") as f:
            f.write(cfg_text)
        mlflow.log_artifact(str(cfg_out), artifact_path="configs")

        feat_text = yaml.safe_dump(feat_spec, sort_keys=True)
        feat_hash = sha1_of_text(feat_text)
        feat_out = artifacts_dir / f"features_{feat_hash}.yaml"
        with open(feat_out, "w") as f:
            f.write(feat_text)
        mlflow.log_artifact(str(feat_out), artifact_path="features")

        # Top-K anomalies table (ids + score)
        k = int(args.topk)
        idx = topk_indices(scores, k)
        out = iddf.iloc[idx].copy()
        out["anomaly_score"] = scores[idx]
        topk_path = artifacts_dir / f"topk_{algo}_{cfg_hash}.csv"
        out.to_csv(topk_path, index=False)
        mlflow.log_artifact(str(topk_path), artifact_path="reports")

        # ---- Optionally log model (sklearn pipelines log cleanly; AE placeholder may not)
        # We skip logging for 'autoencoder' here unless your wrapper is MLflow serializable.
        try:
            if algo in {"isolation_forest", "lof"}:
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path="model",
                    registered_model_name=args.register_model_name
                    if args.register_model_name else None
                )
            # else: leave to separate AE training script/registry if needed
        except Exception as e:
            # Don't fail the run just because model logging failed (e.g., custom wrapper)
            mlflow.set_tag("model_log_error", str(e))

        # ---- Primary metric (for sweeps to read later if needed)
        # You can choose to log a selection metric here; for now we echo the score_mean.
        mlflow.log_metric("primary_metric", stats.get(metric_name, stats["score_mean"]))

        # ---- Summarize to stdout
        print(json.dumps({
            "algo": algo,
            "config_hash": cfg_hash,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "fit_time_s": round(fit_s, 3),
            "metrics": stats,
            "topk_artifact": str(topk_path),
        }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

#==========================================
sweep
#==========================================
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import optuna
import mlflow

DATA_GLOB = "data_stream/processed/date_*/part.parquet"
FEATURES_YAML = "configs/features/fs_v1.yaml"
EXPERIMENT_NAME = "AAA-Experiments"

# Choose the metric your run_experiment prints into the last JSON line
PRIMARY_METRIC = "score_mean"       # or your overlap metric if you log it

def _write_trial_cfg(params: dict, trial_num: int) -> Path:
    cfg_dir = Path("configs/experiments/iforest/_optuna")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / f"trial_{trial_num}.yaml"
    txt = "algo: isolation_forest\nparams:\n" + "\n".join([f"  {k}: {v}" for k,v in params.items()])
    cfg.write_text(txt)
    return cfg

def _run_single(cfg_path: Path) -> dict:
    cmd = [
        sys.executable, "-m", "src.aaa.exp.run_experiment",
        "--data", DATA_GLOB,
        "--features", FEATURES_YAML,
        "--config", str(cfg_path),
        "--experiment-name", EXPERIMENT_NAME,
        "--topk", "200",
    ]
    out = subprocess.check_output(cmd, text=True)
    # last printed line in run_experiment is a JSON summary
    return json.loads(out.splitlines()[-1])

def _suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_samples": trial.suggest_int("max_samples", 256, 4096, step=256),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "contamination": trial.suggest_categorical("contamination", ["auto", 0.01, 0.02]),
        "bootstrap": False,
        "random_state": 42,
    }

def objective(trial: optuna.Trial) -> float:
    params = _suggest_params(trial)
    cfg_path = _write_trial_cfg(params, trial.number)

    # Nested child run for bookkeeping (optional)
    with mlflow.start_run(run_name=f"iforest_optuna_trial_{trial.number}", nested=True):
        mlflow.set_tag("sweep", "iforest_optuna_v1")
        mlflow.log_params(params)
        summary = _run_single(cfg_path)
        score = float(summary["metrics"][PRIMARY_METRIC])
        mlflow.log_metric("objective", score)
        return score

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Optional: persist study so you can pause/continue and parallelize
    # storage = "sqlite:///optuna/iforest.db"
    storage = None

    with mlflow.start_run(run_name="iforest_optuna_parent"):
        mlflow.set_tag("sweep", "iforest_optuna_v1")

        study = optuna.create_study(
            direction="maximize",
            study_name="iforest_bayes_v1",
            storage=storage,
            load_if_exists=bool(storage),
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=25, n_jobs=1)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_objective", float(study.best_value))
        print("Best params:", study.best_params)
        print("Best value:", study.best_value)

if __name__ == "__main__":
    main()
    
# severity_metric.py  (compute-only, reusable in Gold & Inference)
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ---------------------------
# Squash / scaling utilities
# ---------------------------

def _to_num(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def squash_tanh(z: pd.Series, scale: float = 3.0, use_abs: bool = True) -> pd.Series:
    """Smoothly map to [0,1] via tanh(|z|/scale) (or tanh(z/scale) if signed)."""
    z = _to_num(z)
    if use_abs: z = z.abs()
    return np.tanh(z / float(scale))

def squash_logistic(z: pd.Series, scale: float = 3.0, use_abs: bool = True) -> pd.Series:
    """Logistic map to [0,1] with 0.5 near 0; steeper for smaller 'scale'."""
    z = _to_num(z)
    if use_abs: z = z.abs()
    # center at 0; scale controls slope; clip for numerical stability
    x = np.clip(z / float(scale), -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def squash_linear(z: pd.Series, scale: float = 3.0, use_abs: bool = True) -> pd.Series:
    """Linear ramp: min(|z|/scale, 1). Simple and interpretable."""
    z = _to_num(z)
    if use_abs: z = z.abs()
    return (z / float(scale)).clip(0.0, 1.0)

_SQUASHERS = {
    "tanh": squash_tanh,
    "logistic": squash_logistic,
    "linear": squash_linear,
}

# ---------------------------
# Component computation
# ---------------------------

def compute_component_scores(
    df: pd.DataFrame,
    z_cols: Dict[str, str],
    *,
    scales: Optional[Dict[str, float]] = None,
    methods: Optional[Dict[str, str]] = None,
    polarity: Optional[Dict[str, int]] = None,
    use_abs: Optional[Dict[str, bool]] = None,
    component_prefix: str = "",
    default_scale: float = 3.0,
    default_method: str = "tanh",
    default_use_abs: bool = True,
) -> pd.DataFrame:
    """
    Build per-feature component scores in [0,1].

    Parameters
    ----------
    z_cols : mapping {component_name -> df_column_with_raw_signal}
        e.g., {"ZB": "ZB_z_score", "IF": "if_raw_anom"}
    scales : optional per-component scale (e.g., {"IF": 1.2})
    methods : optional per-component squasher {"ZB":"tanh","IF":"logistic","BU":"linear"}
    polarity : optional per-component polarity {name: +1|-1}
        If -1, the signal is multiplied by -1 *before* abs/squash, useful if "higher is better".
    use_abs : optional per-component flag to apply abs() before squashing (default True)
    """
    out = df.copy()
    scales = scales or {}
    methods = methods or {}
    polarity = polarity or {}
    use_abs = use_abs or {}

    for name, col in z_cols.items():
        src = out[col] if col in out.columns else 0.0
        p = int(polarity.get(name, +1))
        src = _to_num(src) * p
        method = methods.get(name, default_method)
        scale = float(scales.get(name, default_scale))
        ua = bool(use_abs.get(name, default_use_abs))

        squasher = _SQUASHERS.get(method)
        if squasher is None:
            raise ValueError(f"Unknown squash method '{method}' for component '{name}'. "
                             f"Valid: {list(_SQUASHERS.keys())}")

        out[f"{component_prefix}{name}_score"] = squasher(src, scale=scale, use_abs=ua)

    return out

# ---------------------------
# Severity computation
# ---------------------------

def compute_severity(
    df: pd.DataFrame,
    *,
    z_cols: Dict[str, str],
    weights: Dict[str, float],
    # optional per-component knobs
    scales: Optional[Dict[str, float]] = None,
    methods: Optional[Dict[str, str]] = None,
    polarity: Optional[Dict[str, int]] = None,
    use_abs: Optional[Dict[str, bool]] = None,
    component_prefix: str = "",
    default_scale: float = 3.0,
    default_method: str = "tanh",
    default_use_abs: bool = True,
    normalize_weights: bool = True,
    # flags & persistence
    flag_cols: Optional[List[str]] = None,
    persistence_col: Optional[str] = None,
    # blend coefficients
    alpha_core: float = 0.85,
    beta_flags: float = 0.10,
    gamma_persist: float = 0.05,
    # outputs
    severity_name: str = "Severity_final",
    label_name: str = "Severity_label",
    label_bins: Tuple[float, float] = (0.30, 0.70),
    return_components: bool = True,
) -> pd.DataFrame:
    """
    Compute severity in [0,1] from engineered signals.

    Works for:
      - Gold layer: pass z_cols & weights only.
      - Inference: pass same z_cols (+ optional model score), weights, and flag_cols.

    Notes:
      - weights can be auto-normalized.
      - per-component scale, method, polarity, use_abs allow independent tuning later.
      - flags are averaged over the PRESENT flag columns and added with beta_flags.
      - persistence is expected in [0,1] (we coerce & clip).
    """
    # Validate keys
    if set(weights.keys()) != set(z_cols.keys()):
        raise AssertionError("weights and z_cols must have identical keys.")

    # Normalize weights (optional)
    if normalize_weights:
        s = sum(float(w) for w in weights.values())
        s = s if s > 0 else 1.0
        weights = {k: float(v) / s for k, v in weights.items()}

    # Component scores
    out = compute_component_scores(
        df,
        z_cols=z_cols,
        scales=scales,
        methods=methods,
        polarity=polarity,
        use_abs=use_abs,
        component_prefix=component_prefix,
        default_scale=default_scale,
        default_method=default_method,
        default_use_abs=default_use_abs,
    )

    # Core blend -> Severity_S0
    s0 = 0.0
    for name in z_cols.keys():
        comp_col = f"{component_prefix}{name}_score"
        s0 += float(weights[name]) * _to_num(out[comp_col])
    out["Severity_S0"] = s0

    # Flags (0/1), normalized by #present columns
    if flag_cols:
        present = [c for c in flag_cols if c in out.columns]
        if present:
            denom = max(1, len(present))
            B = out[present].apply(_to_num).sum(axis=1) / denom
            B = B.clip(0, 1)
        else:
            B = 0.0
    else:
        B = 0.0

    # Persistence in [0,1]
    if persistence_col and persistence_col in out.columns:
        P = _to_num(out[persistence_col]).clip(0, 1)
    else:
        P = 0.0

    # Final severity
    sev = (alpha_core * out["Severity_S0"] +
           beta_flags * B +
           gamma_persist * P)
    out[severity_name] = np.clip(_to_num(sev), 0.0, 1.0)

    # Labels
    lo, hi = label_bins
    def _lab(v: float) -> str:
        if v < lo: return "Low"
        if v < hi: return "Medium"
        return "High"
    out[label_name] = out[severity_name].apply(_lab)

    # Optionally drop component columns to keep output lean
    if not return_components:
        drop_cols = [c for c in out.columns if c.endswith("_score") and c.startswith(component_prefix)]
        out = out.drop(columns=drop_cols, errors="ignore")

    return out
    
# ================================
# GOLD-LAYER PLOTS (returns figs)
# ================================
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# 1) Per-device: overall severity trend
def plot_severity_trend_gold(
    df_sev: pd.DataFrame,
    device_id_val,
    date_col: str = "date",
    severity_col: str = "Severity_final",
    title: str | None = None
) -> Optional[plt.Figure]:
    """
    Build and return a Figure showing severity (0–1) over time for one device.
    Returns None if the device has no rows.
    """
    dd = df_sev[df_sev.get("device_id") == device_id_val].copy()
    if dd.empty:
        return None
    dd = dd.sort_values(date_col)
    x = pd.to_datetime(dd[date_col])
    y = pd.to_numeric(dd[severity_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.axhline(0.3, linestyle="--", linewidth=1)
    ax.axhline(0.7, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Severity trend (Gold) | device_id={device_id_val}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Severity (0–1)")
    ax.grid(True)
    fig.tight_layout()
    return fig


# 2) Per-device: component score trends (from z-scores)
def plot_component_trend_gold(
    df_sev: pd.DataFrame,
    device_id_val,
    component_cols: list[str],
    date_col: str = "date",
    title: str | None = None
) -> Optional[plt.Figure]:
    """
    Build and return a Figure with multiple component score lines (0–1) for one device.
    Returns None if the device has no rows.
    """
    dd = df_sev[df_sev.get("device_id") == device_id_val].copy()
    if dd.empty:
        return None
    dd = dd.sort_values(date_col)
    x = pd.to_datetime(dd[date_col])

    fig, ax = plt.subplots(figsize=(10, 4))
    for c in component_cols:
        if c in dd.columns:
            ax.plot(x, pd.to_numeric(dd[c], errors="coerce"), marker="o", label=c)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Component scores (Gold) | device_id={device_id_val}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Component score (0–1)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# 3) Fleet: daily count of devices above a severity threshold
def plot_daily_severe_device_counts_gold(
    df_sev: pd.DataFrame,
    date_col: str = "date",
    device_col: str = "device_id",
    severity_col: str = "Severity_final",
    threshold: float = 0.7,
    rolling_days: int = 7
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute daily counts of UNIQUE devices with severity >= threshold and
    return (Figure, counts_df).

    counts_df columns: [date, devices_severe, devices_severe_rolling?]
    """
    dd = df_sev.copy()
    dd[date_col] = pd.to_datetime(dd[date_col])
    dd = dd[pd.to_numeric(dd[severity_col], errors="coerce") >= threshold]

    out = (dd.drop_duplicates([date_col, device_col])
             .groupby(date_col, as_index=False)[device_col]
             .nunique()
             .rename(columns={device_col: "devices_severe"}))

    if rolling_days:
        out["devices_severe_rolling"] = (
            out["devices_severe"].rolling(rolling_days, min_periods=1).mean()
        )

    x = pd.to_datetime(out[date_col])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, out["devices_severe"], marker="o", label="Daily devices ≥ threshold")
    if "devices_severe_rolling" in out.columns:
        ax.plot(x, out["devices_severe_rolling"], marker="o", label=f"{rolling_days}-day rolling mean")
    ax.set_title(f"Daily devices with Severity ≥ {threshold} (Gold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unique devices")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, out


# 4) Fleet: daily count of devices with ANY statistical anomaly flag (Gold-only)
def plot_daily_stat_anomaly_counts_gold(
    df_sev: pd.DataFrame,
    z_cols_map: dict[str, str],
    date_col: str = "date",
    device_col: str = "device_id",
    z_threshold: float = 3.0,
    rolling_days: int = 7
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """
    Build a composite statistical flag per row: 1 if ANY |z| >= z_threshold.
    Then compute daily UNIQUE device counts where the flag == 1.

    Returns (Figure, counts_df). If no z-score columns are present, returns (None, None).
    """
    dd = df_sev.copy()
    dd[date_col] = pd.to_datetime(dd[date_col])

    # Build statistical flag (ANY |z| >= threshold)
    present_cols = [c for c in z_cols_map.values() if c in dd.columns]
    if not present_cols:
        return None, None

    Z = dd[present_cols].apply(pd.to_numeric, errors="coerce").abs()
    dd["__stat_is_anom__"] = (Z.ge(z_threshold).any(axis=1)).astype(int)

    flagged = dd[dd["__stat_is_anom__"] == 1]
    out = (flagged.drop_duplicates([date_col, device_col])
                   .groupby(date_col, as_index=False)[device_col]
                   .nunique()
                   .rename(columns={device_col: "devices_flagged"}))

    if rolling_days:
        out["devices_flagged_rolling"] = (
            out["devices_flagged"].rolling(rolling_days, min_periods=1).mean()
        )

    x = pd.to_datetime(out[date_col])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, out["devices_flagged"], marker="o", label="Daily devices (|z|≥thr)")
    if "devices_flagged_rolling" in out.columns:
        ax.plot(x, out["devices_flagged_rolling"], marker="o", label=f"{rolling_days}-day rolling mean")
    ax.set_title(f"Daily devices flagged by statistical rule |z| ≥ {z_threshold} (Gold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unique devices")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, out
# =====================================
# AAA Severity Configuration
# =====================================

# ---------- COMPUTE ----------
severity:
  tanh_scale: 3.0               # scale for tanh(|z|/scale)
  label_bins: [0.30, 0.70]      # thresholds for Low/Medium/High severity
  normalize_weights: true       # normalize weights automatically

# ---------- FEATURE MAPPING ----------
z_cols:
  SC: SC_z_score_session_cnt
  SS: SS_z_score_max
  SL: SL_z_score
  ZB: ZB_z_score
  BU: BU_z_score_bytes_usage
  IDLE: IDLE_z_idle

# ---------- WEIGHTS ----------
weights:
  SC: 0.15
  SS: 0.15
  SL: 0.20
  ZB: 0.20
  BU: 0.15
  IDLE: 0.15

# ---------- FLAGS ----------
# For inference module, use these (ignored in Gold)
flag_cols:
  - anomaly
  - shortSessionAnomaly
  - ZerobytesUsagenAnomaly

# ---------- PERSISTENCE ----------
persistence_col: null

# ---------- PLOTTING ----------
plotting:
  do_plots: true
  severity_threshold: 0.7
  rolling_days: 7

# to integrate model-specific weights or AE configs
inference:
  include_model_score: true
  model_score_col: iforest_score
  model_weight: 0.08


# severity/severity_handler.py
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd

from severity.severity_metric import compute_severity

def run_severity_handler(
    df: pd.DataFrame,
    config_path: str | None = None,
    # overrides (take priority over config)
    z_cols: dict | None = None,
    weights: dict | None = None,
    flag_cols: list[str] | None = None,
    persistence_col: str | None = None,
    # id fields / names
    date_col: str = "date",
    device_col: str = "device_id",
    severity_col: str = "Severity_final",
    label_col: str = "Severity_label",
    # compute knobs
    tanh_scale: float | None = None,
    label_bins: tuple[float, float] | None = None,
    # plotting knobs
    do_plots: bool | None = None,
    return_figs: bool = False,          # <— NEW
    severity_threshold: float | None = None,
    rolling_days: int | None = None,
    plotters: dict | None = None,       # dict of plotting functions
):
    """
    Compute severity from Gold/Inference features and optionally generate plots.

    Returns a dict:
      {
        "df_sev": <DataFrame>,
        "daily_anom": <DataFrame | None>,
        "daily_sev": <DataFrame>,
        "figs": {"severity_trend": fig, "component_trend": fig,
                 "daily_sev": fig, "daily_stat": fig_or_None}  # only if return_figs=True
      }
    """
    # -----------------
    # 1) Load config
    # -----------------
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    z_cols = z_cols or cfg.get("z_cols", {})
    weights = weights or cfg.get("weights", {})
    flag_cols = flag_cols or cfg.get("flag_cols", None)
    persistence_col = persistence_col or cfg.get("persistence_col", None)

    severity_cfg = cfg.get("severity", {})
    tanh_scale = tanh_scale if tanh_scale is not None else severity_cfg.get("tanh_scale", 3.0)
    lb = severity_cfg.get("label_bins", [0.30, 0.70])
    label_bins = label_bins or (float(lb[0]), float(lb[1]))

    plot_cfg = cfg.get("plotting", {})
    if do_plots is None:
        do_plots = bool(plot_cfg.get("do_plots", True))
    severity_threshold = (severity_threshold
                          if severity_threshold is not None
                          else float(plot_cfg.get("severity_threshold", 0.7)))
    rolling_days = (rolling_days
                    if rolling_days is not None
                    else int(plot_cfg.get("rolling_days", 7)))

    # -----------------
    # 2) Compute
    # -----------------
    df_sev = compute_severity(
        df=df,
        z_cols=z_cols,
        weights=weights,
        flag_cols=flag_cols,
        persistence_col=persistence_col,
        scale=tanh_scale,
        severity_name=severity_col,
        label_name=label_col,
        label_bins=label_bins,
    )

    results: dict = {"df_sev": df_sev, "daily_anom": None, "daily_sev": None, "figs": {}}

    # -----------------
    # 3) Plots (optional)
    # -----------------
    if do_plots and plotters:
        # pick a device to plot
        device_id_to_plot = df_sev[device_col].iloc[0] if device_col in df_sev.columns and not df_sev.empty else None

        # per-device severity trend
        if "plot_severity_trend_gold" in plotters and device_id_to_plot is not None:
            fig = plotters["plot_severity_trend_gold"](
                df_sev, device_id_to_plot, date_col=date_col, severity_col=severity_col
            )
            if return_figs and fig is not None:
                results["figs"]["severity_trend"] = fig

        # per-device component trends
        if "plot_component_trend_gold" in plotters and device_id_to_plot is not None:
            comp_cols = [f"{k}_score" for k in z_cols.keys()]
            fig = plotters["plot_component_trend_gold"](
                df_sev, device_id_to_plot, component_cols=comp_cols, date_col=date_col
            )
            if return_figs and fig is not None:
                results["figs"]["component_trend"] = fig

        # fleet: daily devices above severity threshold
        if "plot_daily_severe_device_counts_gold" in plotters:
            fig, daily_sev = plotters["plot_daily_severe_device_counts_gold"](
                df_sev, date_col=date_col, device_col=device_col,
                severity_col=severity_col, threshold=severity_threshold,
                rolling_days=rolling_days
            )
            results["daily_sev"] = daily_sev
            if return_figs and fig is not None:
                results["figs"]["daily_sev"] = fig

        # fleet: daily devices flagged by statistical rule (if you want this view)
        if "plot_daily_stat_anomaly_counts_gold" in plotters and z_cols:
            fig, daily_anom = plotters["plot_daily_stat_anomaly_counts_gold"](
                df_sev, z_cols_map=z_cols, date_col=date_col, device_col=device_col,
                z_threshold=3.0, rolling_days=rolling_days
            )
            results["daily_anom"] = daily_anom
            if return_figs and fig is not None:
                results["figs"]["daily_stat"] = fig

    return results
    
from src.aaa.exp.severity.severity_handler import run_severity_handler
from src.aaa.exp.severity.severity_plots import (
    plot_severity_trend_gold,
    plot_component_trend_gold,
    plot_daily_severe_device_counts_gold,
    plot_daily_stat_anomaly_counts_gold,
)

plotters = {
    "plot_severity_trend_gold": plot_severity_trend_gold,
    "plot_component_trend_gold": plot_component_trend_gold,
    "plot_daily_severe_device_counts_gold": plot_daily_severe_device_counts_gold,
    "plot_daily_stat_anomaly_counts_gold": plot_daily_stat_anomaly_counts_gold,
}

results = run_severity_handler(
    df=feat,                                    # your Gold-layer dataframe
    config_path="severity/severity_config.yaml",
    plotters=plotters,
    do_plots=True,
    return_figs=True,                            # <-- now supported
)

df_sev   = results["df_sev"]
daily_an = results["daily_anom"]
daily_sv = results["daily_sev"]
figs     = results["figs"]

# show inline
for name, fig in (figs or {}).items():
    if fig:
        print(f"Showing: {name}")
        fig.show()