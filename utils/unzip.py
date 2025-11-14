import zipfile
import os

# path to your zip file
zip_path = "code/generation/stage_2/AdvCloak.zip"

# extract to the same directory
extract_dir = os.path.dirname(zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted to: {extract_dir}")