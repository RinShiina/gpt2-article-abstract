from huggingface_hub import HfApi

def upload_model(model_path, commit_message):
    api = HfApi()
    api.upload_folder(
        path_or_fileobj=model_path,
        repo_id="", # Specify your Hugging Face repo ID here
        commit_message=commit_message,
        repo_type="model"
    )