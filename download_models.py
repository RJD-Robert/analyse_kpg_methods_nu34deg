#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def download_model(model_name: str, local_dir: str):
    """
    Download a model from Hugging Face and save it to the specified local directory.
    """
    try:
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print(f"Model {model_name} downloaded successfully to {local_dir}.")
    except HfHubHTTPError as e:
        print(f"Failed to download model {model_name}: {e}")


if __name__ == "__main__":
    models = {
        # "BAAI/bge-large-en-v1.5": "/Volumes/storage-webisstud/data-tmp/current/nu34deg/src/codereplication/li2024/models/bge-large-en-v1.5"
        # "webis/argument-quality-ibm-reproduced": "/Users/robertjosef/development/codereplication_nu34deg/alshomary2021/models/argument-quality-ibm-reproduced",
        # "sentence-transformers/all-mpnet-base-v2": "/Users/robertjosef/development/codereplication_nu34deg/evaluation/models/all-mpnet-base-v2 ",
        # "FacebookAI/roberta-large-mnli": "/mnt/ceph/storage/data-tmp/current/nu34deg/src/codereplication/evaluation/models/roberta-large-mnli"
        # "cross-encoder/stsb-roberta-large": "/mnt/ceph/storage/data-tmp/current/nu34deg/src/codereplication/evaluation/models/stsb-roberta-large"
    }

    for model_name, local_dir in models.items():
        os.makedirs(local_dir, exist_ok=True)
        download_model(model_name, local_dir)

    print("All models downloaded successfully.")
