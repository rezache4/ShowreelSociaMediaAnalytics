# CELL 1: Extract undownloaded image and carousel links from GCS bucket
# This cell extracts which images/carousels have NOT been downloaded yet

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
IG_POSTS_PATH = "Data/Camihawke/Cleaned/ig_posts_with_duration.parquet"
IMAGE_BUCKET_PATH = "multimodal_dataset_fixed/image/"
CAROUSEL_BUCKET_PATH = "multimodal_dataset_fixed/carousel/"
OUTPUT_LINKS_FILE = "undownloaded_links.csv"

# Initialize GCS client
storage_client = storage.Client(project=GCS_PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

print("="*70)
print("EXTRACTING UNDOWNLOADED IMAGE AND CAROUSEL LINKS")
print("="*70)

# Load ig_posts from bucket
print(f"\n[1/5] Loading ig_posts from: gs://{BUCKET_NAME}/{IG_POSTS_PATH}")
try:
    blob = bucket.blob(IG_POSTS_PATH)
    ig_posts = pd.read_parquet(blob.open('rb'))
    print(f"✓ Loaded {len(ig_posts)} posts")
except Exception as e:
    print(f"✗ Error loading ig_posts: {e}")
    raise

# Get image and carousel links
print(f"\n[2/5] Extracting links from ig_posts...")
image_links = ig_posts[ig_posts["media_type"] == "IMAGE"]["permalink"].tolist()
carousel_links = ig_posts[ig_posts["media_type"] == "CAROUSEL_ALBUM"]["permalink"].tolist()

print(f"✓ Found {len(image_links)} IMAGE posts")
print(f"✓ Found {len(carousel_links)} CAROUSEL_ALBUM posts")

# List existing downloaded files in bucket
def list_downloaded_shortcodes(bucket, prefix):
    """Extract shortcodes of already downloaded posts"""
    shortcodes = set()
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # Path format: multimodal_dataset_fixed/image/{shortcode}/{files}
        parts = blob.name[len(prefix):].split('/')
        if parts[0] and parts[0].strip():  # Extract shortcode
            shortcodes.add(parts[0])

    return shortcodes

print(f"\n[3/5] Checking already downloaded files in bucket...")
already_downloaded_images = list_downloaded_shortcodes(bucket, IMAGE_BUCKET_PATH)
already_downloaded_carousels = list_downloaded_shortcodes(bucket, CAROUSEL_BUCKET_PATH)

print(f"✓ Already downloaded images: {len(already_downloaded_images)}")
print(f"✓ Already downloaded carousels: {len(already_downloaded_carousels)}")

# Extract shortcode from permalink
def extract_shortcode(permalink):
    """Extract Instagram shortcode from permalink"""
    return permalink.split('/')[-2] if '/' in permalink else permalink

# Filter undownloaded links
undownloaded_images = [
    url for url in image_links
    if extract_shortcode(url) not in already_downloaded_images
]
undownloaded_carousels = [
    url for url in carousel_links
    if extract_shortcode(url) not in already_downloaded_carousels
]

print(f"\n[4/5] Undownloaded items:")
print(f"✓ Undownloaded images: {len(undownloaded_images)}")
print(f"✓ Undownloaded carousels: {len(undownloaded_carousels)}")
print(f"✓ Total to download: {len(undownloaded_images) + len(undownloaded_carousels)}")

# Create DataFrame with undownloaded links
undownloaded_data = []
for url in undownloaded_images:
    undownloaded_data.append({"url": url, "type": "IMAGE"})
for url in undownloaded_carousels:
    undownloaded_data.append({"url": url, "type": "CAROUSEL_ALBUM"})

undownloaded_df = pd.DataFrame(undownloaded_data)

# Save to bucket as CSV
print(f"\n[5/5] Saving undownloaded links...")
csv_content = undownloaded_df.to_csv(index=False)
blob = bucket.blob(OUTPUT_LINKS_FILE)
blob.upload_from_string(csv_content, content_type='text/csv')
print(f"✓ Saved to: gs://{BUCKET_NAME}/{OUTPUT_LINKS_FILE}")

# Also save locally for reference
undownloaded_df.to_csv("undownloaded_links.csv", index=False)
print(f"✓ Local copy: undownloaded_links.csv")

print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
print(f"Next step: Run CELL 2 to download the {len(undownloaded_df)} links")
