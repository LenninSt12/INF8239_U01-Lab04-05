import os
import hashlib
from dotenv import load_dotenv

load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

print("Verificando integridad del dataset local...")
sha256_hash = hashlib.sha256()
with open(dataset_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)

print(f"Hash SHA-256: {sha256_hash.hexdigest()}")