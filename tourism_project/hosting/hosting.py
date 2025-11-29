from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing deployment files
    repo_id="hsaluja431/TourismProject",          # the target repo of hugging face
    repo_type="space",                      #space where the deployment files will be uploaded
    path_in_repo="",                        
)
