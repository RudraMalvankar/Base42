import os
from huggingface_hub import hf_hub_download

# Using Gemma-2-2B-it-GGUF to qualify for the $6,000 DeepMind Gemma Prize!
# Q4_K_M fits well within the 4GB RAM limit.
REPO_ID = "bartowski/gemma-2-2b-it-GGUF"
FILENAME = "gemma-2-2b-it-Q4_K_M.gguf"
MODEL_DIR = "/model"

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {FILENAME} from {REPO_ID} (Targeting Gemma Prize)...")
    
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
