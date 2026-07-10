import os
import sys
import urllib.request
from utils.logger import setup_logger

logger = setup_logger("download_model")

# We use Qwen2.5-1.5B-Instruct-GGUF quantized in Q4_K_M (size: ~1.2 GB)
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUTPUT_DIR = "models"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "local_model.gguf")

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = (read_so_far * 100) / total_size
        sys.stdout.write(f"\rDownloading: {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading: {read_so_far / (1024*1024):.2f} MB")
        sys.stdout.flush()

def main():
    if os.path.exists(OUTPUT_PATH):
        logger.info(f"Model already exists at '{OUTPUT_PATH}'. Skipping download.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Starting download from: {MODEL_URL}")
    logger.info(f"Saving to: {OUTPUT_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, OUTPUT_PATH, download_progress)
        print()  # Newline after progress bar
        logger.info("Download completed successfully!")
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        # Clean up partial download if it exists
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        sys.exit(1)

if __name__ == "__main__":
    main()
