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
