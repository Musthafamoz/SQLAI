from huggingface_hub import snapshot_download


model_id="q"
snapshot_download(repo_id=model_id, local_dir="",
                  local_dir_use_symlinks=False, revision="main", hf_token="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")