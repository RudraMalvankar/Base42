import os
from huggingface_hub import hf_hub_download

# Using Qwen2.5-1.5B-Instruct for superior instruction following and factual accuracy.
# Q4_K_M quantization (~0.99 GB) fits well within the 4GB RAM limit.
REPO_ID = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
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
