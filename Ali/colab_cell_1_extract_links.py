# CELL 1: Extract undownloaded links using downloaded_links.csv as reference
# This cell uses your local downloaded_links.csv to determine what's missing

# Install required libraries
import subprocess
import sys

libs_to_install = ['google-cloud-storage', 'pandas', 'tqdm']
for lib in libs_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", lib])

import os
import io
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
DOWNLOADED_LINKS_FILE = "downloaded_links.csv"  # Upload this from local notebook
OUTPUT_LINKS_FILE = "undownloaded_links.csv"

# Initialize GCS client
storage_client = storage.Client(project=GCS_PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

print("="*70)
print("EXTRACTING UNDOWNLOADED IMAGE AND CAROUSEL LINKS")
print("="*70)

# ============================================================================
# Step 1: Load downloaded_links.csv (from local notebook)
# ============================================================================
print(f"\n[1/4] Loading downloaded links from: {DOWNLOADED_LINKS_FILE}")
try:
    downloaded_df = pd.read_csv(DOWNLOADED_LINKS_FILE)
    downloaded_shortcodes = set(downloaded_df['shortcode'].tolist())
    print(f"✓ Loaded {len(downloaded_df)} already downloaded items")
    print(f"  • IMAGE: {len(downloaded_df[downloaded_df['type'] == 'IMAGE'])}")
    print(f"  • CAROUSEL_ALBUM: {len(downloaded_df[downloaded_df['type'] == 'CAROUSEL_ALBUM'])}")
except FileNotFoundError:
    print(f"✗ Error: {DOWNLOADED_LINKS_FILE} not found!")
    print("Make sure you uploaded downloaded_links.csv from your local notebook to Colab Files")
    raise
except Exception as e:
    print(f"✗ Error loading CSV: {e}")
    raise

# ============================================================================
# Step 2: Load ig_posts from bucket
# ============================================================================
print(f"\n[2/4] Loading ig_posts from: gs://{BUCKET_NAME}/{IG_POSTS_PATH}")
try:
    blob = bucket.blob(IG_POSTS_PATH)
    ig_posts = pd.read_parquet(blob.open('rb'))
    print(f"✓ Loaded {len(ig_posts)} total posts")
    print(f"  • IMAGE: {len(ig_posts[ig_posts['media_type'] == 'IMAGE'])}")
    print(f"  • CAROUSEL_ALBUM: {len(ig_posts[ig_posts['media_type'] == 'CAROUSEL_ALBUM'])}")
except Exception as e:
    print(f"✗ Error loading ig_posts: {e}")
    raise

# ============================================================================
# Step 3: Extract shortcode from permalink and compare
# ============================================================================
print(f"\n[3/4] Finding undownloaded items...")

def extract_shortcode(permalink):
    """Extract Instagram shortcode from permalink"""
    return permalink.split('/')[-2] if '/' in permalink else permalink

# Add shortcode column to ig_posts
ig_posts['shortcode_extracted'] = ig_posts['permalink'].apply(extract_shortcode)

# Filter for undownloaded items (not in downloaded_shortcodes)
undownloaded_df = ig_posts[
    ~ig_posts['shortcode_extracted'].isin(downloaded_shortcodes)
][['permalink', 'media_type']].copy()

# Rename columns to match expected format
undownloaded_df = undownloaded_df.rename(columns={
    'permalink': 'url',
    'media_type': 'type'
})

image_count = len(undownloaded_df[undownloaded_df['type'] == 'IMAGE'])
carousel_count = len(undownloaded_df[undownloaded_df['type'] == 'CAROUSEL_ALBUM'])

print(f"✓ Undownloaded items found:")
print(f"  • IMAGE: {image_count}")
print(f"  • CAROUSEL_ALBUM: {carousel_count}")
print(f"  • Total to download: {len(undownloaded_df)}")

# ============================================================================
# Step 4: Save undownloaded links to bucket
# ============================================================================
print(f"\n[4/4] Saving undownloaded links to bucket...")
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
print(f"\nSummary:")
print(f"  • Already downloaded: {len(downloaded_df)}")
print(f"  • Total in ig_posts: {len(ig_posts)}")
print(f"  • Remaining to download: {len(undownloaded_df)}")
print(f"\nNext step: Run CELL 2 to download the {len(undownloaded_df)} remaining items")
