import os
from huggingface_hub import HfApi
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
api = HfApi()
print("Uploading data.....")
api.upload_folder(
    folder_path=str(PROJECT_ROOT / "data"),
    repo_id="satishjasthij/Unsplash-Visual-Semantic",
    repo_type="space",
    token=os.getenv("HUGGINGFACE_TOKEN"),
    commit_message="add dataset",
    create_pr=True,
)
print("Finished uploading data")