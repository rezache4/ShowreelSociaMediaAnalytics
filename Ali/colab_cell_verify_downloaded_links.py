# CELL 3 (Optional): Verify downloaded links in GCS and update ig_posts with download status

# Install required libraries
import subprocess
import sys

libs_to_install = ['google-cloud-storage', 'pandas', 'tqdm']
for lib in libs_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", lib])

import os
import io
from pathlib import Path
from google.cloud import storage
from google.colab import auth
import pandas as pd

# Authenticate with Google Cloud
print("Authenticating with Google Cloud...")
auth.authenticate_user()

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================
BUCKET_NAME = "your-bucket-name"  # ← CHANGE THIS
GCS_PROJECT_ID = "your-project-id"  # ← CHANGE THIS
DOWNLOADED_LINKS_FILE = "downloaded_links.csv"  # Upload this file from local notebook
IMAGE_BUCKET_PATH = "multimodal_dataset_fixed/image/"
CAROUSEL_BUCKET_PATH = "multimodal_dataset_fixed/carousel/"

# Initialize GCS client
storage_client = storage.Client(project=GCS_PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

print("="*70)
print("VERIFYING DOWNLOADED LINKS IN GCS")
print("="*70)

# ============================================================================
# Step 1: Load the downloaded links CSV
# ============================================================================
print(f"\n[1/3] Loading downloaded links from: {DOWNLOADED_LINKS_FILE}")
try:
    downloaded_df = pd.read_csv(DOWNLOADED_LINKS_FILE)
    print(f"✓ Loaded {len(downloaded_df)} downloaded items\n")
except FileNotFoundError:
    print(f"✗ Error: {DOWNLOADED_LINKS_FILE} not found!")
    print("Upload the downloaded_links.csv from your local notebook to Colab Files first")
    raise

# ============================================================================
# Step 2: Verify each downloaded item exists in GCS bucket
# ============================================================================
print("[2/3] Verifying files in GCS bucket...")

verified_count = 0
missing_count = 0
missing_files = []

for idx, row in downloaded_df.iterrows():
    shortcode = row['shortcode']
    post_type = row['type']

    # Determine GCS path
    if post_type == "IMAGE":
        bucket_path = f"{IMAGE_BUCKET_PATH}{shortcode}/"
    else:
        bucket_path = f"{CAROUSEL_BUCKET_PATH}{shortcode}/"

    # Check if files exist in bucket
    blobs = list(bucket.list_blobs(prefix=bucket_path, max_results=1))

    if blobs:
        verified_count += 1
    else:
        missing_count += 1
        missing_files.append(row.to_dict())

    # Progress indicator
    if (idx + 1) % 100 == 0:
        print(f"  Checked {idx + 1}/{len(downloaded_df)}...")

print(f"\n✓ Verification complete:")
print(f"  • Verified in GCS: {verified_count}")
print(f"  • Missing from GCS: {missing_count}")
print(f"  • Verification rate: {(verified_count/len(downloaded_df)*100):.1f}%")

# ============================================================================
# Step 3: Save report and missing items
# ============================================================================
print(f"\n[3/3] Generating reports...")

# Save verification report
report = {
    "total_items": len(downloaded_df),
    "verified": verified_count,
    "missing": missing_count,
    "verification_rate": f"{(verified_count/len(downloaded_df)*100):.1f}%"
}

print("\nVerification Report:")
for key, value in report.items():
    print(f"  • {key}: {value}")

# Save missing items if any
if missing_files:
    missing_df = pd.DataFrame(missing_files)
    csv_content = missing_df.to_csv(index=False)
    blob = bucket.blob("missing_from_bucket.csv")
    blob.upload_from_string(csv_content, content_type='text/csv')

    missing_df.to_csv("missing_from_bucket.csv", index=False)
    print(f"\n⚠ Missing items saved to:")
    print(f"  - GCS: gs://{BUCKET_NAME}/missing_from_bucket.csv")
    print(f"  - Local: missing_from_bucket.csv")
else:
    print("\n✓ All downloaded links verified in GCS bucket!")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
