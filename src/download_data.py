import os
import shutil
import kagglehub

def main():
    print("Downloading dataset 'suvroo/amazon-ml' via kagglehub...")
    path = kagglehub.dataset_download("suvroo/amazon-ml")
    print(f"Downloaded to cache: {path}")

    target_dir = os.path.abspath("amazon_ml_challenge/data")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Symlinking/copying files into: {target_dir}")

    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(d):
            print(f"Already exists: {item}")
            continue
        if os.path.isdir(s):
            # create symlink or copy
            try:
                os.symlink(s, d)
                print(f"Symlinked directory: {item}")
            except Exception:
                shutil.copytree(s, d)
                print(f"Copied directory: {item}")
        else:
            try:
                os.symlink(s, d)
                print(f"Symlinked file: {item}")
            except Exception:
                shutil.copy2(s, d)
                print(f"Copied file: {item}")

    print("\nDataset ready in amazon_ml_challenge/data:")
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        size = os.path.getsize(fp) if os.path.isfile(fp) else "DIR"
        print(f" - {f} ({size} bytes)")

if __name__ == "__main__":
    main()
