import os
import httpx

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
WEIGHTS_DIR = "./weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "model.gguf")

def download():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print("Model already exists. Skipping download.")
        return
        
    print(f"Downloading model from {MODEL_URL}...")
    with httpx.stream("GET", MODEL_URL, follow_redirects=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    download()
