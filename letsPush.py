from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./mon_modele_local",
    repo_id="YesserYes/NoobProject",  # Remplace par ton username/repo
    repo_type="model"
)