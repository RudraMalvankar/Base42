import os
from huggingface_hub import hf_hub_download

# Using Qwen 2.5 1.5B Instruct Q4 (Requires < 1.5GB RAM)
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_DIR = "/model"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {FILENAME} from {REPO_ID}...")
    
    # hf_hub_download automatically handles retries and resuming
    download_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME, 
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded to: {download_path}")

if __name__ == "__main__":
    main()
