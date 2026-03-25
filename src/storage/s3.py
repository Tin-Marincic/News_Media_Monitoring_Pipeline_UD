import boto3
from botocore.exceptions import NoCredentialsError
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.logger import logging

S3_ENDPOINT = "http://localhost:4566"
S3_BUCKET_NAME = "news-pipeline-bucket"


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


def upload_file_to_s3(file_path, file_name):
    s3 = create_s3_client()

    try:
        logging.info(f"Started uploading {file_name} to S3 bucket {S3_BUCKET_NAME}")
        s3.upload_file(file_path, S3_BUCKET_NAME, file_name)
        logging.info(f"Successfully uploaded {file_name} to {S3_BUCKET_NAME}")

    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
    except NoCredentialsError:
        logging.error("Credentials not available.")
    except Exception as e:
        logging.error(f"Unexpected error uploading {file_name}: {e}")


if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "api"))

    for file in os.listdir(base_path):
        if file.endswith(".json"):
            file_path = os.path.join(base_path, file)
            upload_file_to_s3(file_path, file)