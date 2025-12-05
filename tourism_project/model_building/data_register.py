from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Hugging Face Repo Id where the dataset will be pushed
repo_id = "hsaluja431/TourismProject"
# As we are pushing dataset hence the repo_type will be dataset
repo_type = "dataset"

# Initialize API client with HF_TOKEN in the environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Checking if the Dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

#Uploading the csv file which is loaded in the data folder to hugging face
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
