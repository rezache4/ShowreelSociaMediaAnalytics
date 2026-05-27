# CELL 2: Standalone Instagram download pipeline for Google Colab Enterprise
# This cell downloads undownloaded images/carousels and saves directly to GCS

# Install all required libraries
import subprocess
import sys

required_libs = [
    'instaloader',
    'google-cloud-storage',
    'pandas',
    'tqdm',
    'python-dotenv',
    'Pillow'
]

print("Installing required libraries...")
for lib in required_libs:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", lib])
    except:
        pass  # Continue even if one fails
print("✓ All libraries installed\n")

import os
import io
import time
import random
import shutil
from pathlib import Path
from google.cloud import storage
from google.colab import auth
import pandas as pd
import instaloader
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Authenticate with Google Cloud
print("Authenticating with Google Cloud...")
auth.authenticate_user()

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================
BUCKET_NAME = "your-bucket-name"  # ← CHANGE THIS
GCS_PROJECT_ID = "your-project-id"  # ← CHANGE THIS
UNDOWNLOADED_LINKS_FILE = "undownloaded_links.csv"
IMAGE_BUCKET_PATH = "multimodal_dataset_fixed/image/"
CAROUSEL_BUCKET_PATH = "multimodal_dataset_fixed/carousel/"
FAILED_LINKS_FILE = "failed_downloads.csv"

# ============================================================================
# Instagram Credentials - Get from environment/Colab secrets
# ============================================================================
# Option 1: Environment variables
IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")

# Option 2: Colab Secrets (if available)
try:
    from google.colab import userdata
    if not IG_USERNAME:
        IG_USERNAME = userdata.get("IG_USERNAME")
    if not IG_PASSWORD:
        IG_PASSWORD = userdata.get("IG_PASSWORD")
except:
    pass

# Option 3: .env file (if uploaded to Colab)
if not IG_USERNAME or not IG_PASSWORD:
    try:
        load_dotenv(".env")
        IG_USERNAME = IG_USERNAME or os.getenv("IG_USERNAME")
        IG_PASSWORD = IG_PASSWORD or os.getenv("IG_PASSWORD")
    except:
        pass

if not IG_USERNAME:
    print("⚠ IG_USERNAME not found!")
    print("Please provide credentials in one of these ways:")
    print("1. Set environment variables: IG_USERNAME and IG_PASSWORD")
    print("2. Create Colab Secrets named: IG_USERNAME and IG_PASSWORD")
    print("3. Upload .env file with credentials")
    raise ValueError("Instagram credentials required")

print(f"✓ Using Instagram account: {IG_USERNAME}")
print(f"✓ Password configured: {'Yes' if IG_PASSWORD else 'No (will use interactive login)'}\n")

# Initialize GCS client
storage_client = storage.Client(project=GCS_PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# ============================================================================
# Human-like pause function with jitter
# ============================================================================
def human_like_pause(min_seconds=2, max_seconds=8, variation=0.3):
    """Add realistic human-like pause with jitter to avoid detection"""
    base_pause = random.uniform(min_seconds, max_seconds)
    if random.random() < 0.10:  # 10% chance of longer pause
        base_pause *= random.uniform(1.5, 2.5)
    jitter = random.gauss(0, base_pause * variation)
    pause_time = max(0.5, base_pause + jitter)
    return pause_time

# ============================================================================
# Load undownloaded links from bucket
# ============================================================================
print("="*70)
print("LOADING DOWNLOAD QUEUE")
print("="*70)
print(f"\nLoading undownloaded links from: gs://{BUCKET_NAME}/{UNDOWNLOADED_LINKS_FILE}")

try:
    blob = bucket.blob(UNDOWNLOADED_LINKS_FILE)
    undownloaded_csv = blob.download_as_string().decode('utf-8')
    undownloaded_df = pd.read_csv(io.StringIO(undownloaded_csv))
    print(f"✓ Loaded {len(undownloaded_df)} links to download\n")

    if len(undownloaded_df) == 0:
        print("✓ No links to download. Everything is up to date!")
        import sys
        sys.exit(0)

except Exception as e:
    print(f"✗ Error loading links: {e}")
    print("Make sure you ran CELL 1 first to extract undownloaded links!")
    raise

# ============================================================================
# Initialize Instaloader
# ============================================================================
print("="*70)
print("INITIALIZING INSTAGRAM DOWNLOADER")
print("="*70)
print("\nInitializing Instaloader...")

L = instaloader.Instaloader(
    download_pictures=True,
    download_videos=False,
    download_video_thumbnails=False,
    download_geotags=False,
    download_comments=False,
    save_metadata=True,
    compress_json=False,
    post_metadata_txt_pattern="{date_utc}_UTC",
    max_connection_attempts=3,
)

# ============================================================================
# Authenticate with Instagram
# ============================================================================
print("Authenticating with Instagram...")
try:
    L.load_session_from_file(IG_USERNAME)
    print(f"✓ Loaded cached session for {IG_USERNAME}\n")
except FileNotFoundError:
    print(f"Session not found. Logging in...")
    try:
        if IG_PASSWORD:
            L.login(IG_USERNAME, IG_PASSWORD)
            print(f"✓ Logged in with credentials\n")
        else:
            print("Enter your Instagram password in the prompt below:")
            L.interactive_login(IG_USERNAME)
            print()
        L.save_session_to_file()
    except Exception as e:
        print(f"✗ Login failed: {e}")
        raise

# ============================================================================
# Upload function for GCS
# ============================================================================
def upload_directory_to_gcs(local_dir, bucket_obj, gcs_prefix):
    """Upload entire directory to GCS"""
    local_path = Path(local_dir)

    for item in local_path.rglob('*'):
        if item.is_file():
            rel_path = item.relative_to(local_path.parent)
            gcs_full_path = f"{gcs_prefix}{rel_path}".replace("\\", "/")
            blob = bucket_obj.blob(gcs_full_path)
            blob.upload_from_filename(str(item))

# ============================================================================
# Download loop
# ============================================================================
print("="*70)
print("STARTING DOWNLOAD PIPELINE")
print("="*70 + "\n")

failed_urls = []
successful_downloads = 0
download_dir = Path("/tmp/ig_download")
download_dir.mkdir(exist_ok=True, parents=True)

for idx, row in enumerate(undownloaded_df.itertuples(), 1):
    try:
        url = row.url
        post_type = row.type
        shortcode = url.split('/')[-2]

        # Determine target bucket path
        if post_type == "IMAGE":
            target_bucket_path = f"{IMAGE_BUCKET_PATH}{shortcode}/"
        else:
            target_bucket_path = f"{CAROUSEL_BUCKET_PATH}{shortcode}/"

        # Check if already in bucket (in case of concurrent runs)
        existing = list(bucket.list_blobs(prefix=target_bucket_path, max_results=1))
        if existing:
            print(f"[{idx}/{len(undownloaded_df)}] ✓ Already in bucket: {shortcode}")
            continue

        # Download locally first
        local_dir = download_dir / shortcode
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(exist_ok=True, parents=True)

        print(f"[{idx}/{len(undownloaded_df)}] ⬇️ Downloading {post_type}: {shortcode}...")

        # Download post
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target=local_dir)

        # Clean up unnecessary files (keep only media files)
        for item in local_dir.glob('*'):
            if item.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.mp4', '.json']:
                try:
                    item.unlink()
                except:
                    pass

        # Upload to GCS
        print(f"  ⬆️ Uploading to GCS...")
        upload_directory_to_gcs(local_dir, bucket, target_bucket_path)

        # Clean up local directory
        shutil.rmtree(local_dir)

        successful_downloads += 1
        print(f"  ✓ Success!")

        # Human-like pause between downloads
        pause_duration = human_like_pause(min_seconds=3, max_seconds=12, variation=0.35)
        print(f"  ⏸ Pausing {pause_duration:.2f}s (human-like delay)\n")
        time.sleep(pause_duration)

    except instaloader.exceptions.LoginRequiredException:
        print(f"⚠ Login expired. Re-authenticating...")
        try:
            if IG_PASSWORD:
                L.login(IG_USERNAME, IG_PASSWORD)
            else:
                L.interactive_login(IG_USERNAME)
            L.save_session_to_file()
            print("✓ Re-authenticated successfully")

            # Retry download
            pause_duration = human_like_pause(min_seconds=5, max_seconds=15, variation=0.4)
            print(f"  ⏸ Pausing {pause_duration:.2f}s before retry\n")
            time.sleep(pause_duration)

        except Exception as auth_exc:
            print(f"✗ Re-authentication failed: {auth_exc}")
            failed_urls.append(url)

    except Exception as e:
        print(f"✗ Failed to download {shortcode}: {str(e)[:100]}")
        failed_urls.append(url)
        time.sleep(human_like_pause(min_seconds=2, max_seconds=5))

# Clean up download directory
if download_dir.exists():
    shutil.rmtree(download_dir)

# ============================================================================
# Save results
# ============================================================================
print("="*70)
print("DOWNLOAD PIPELINE COMPLETE")
print("="*70)

summary = {
    "total_processed": len(undownloaded_df),
    "successful": successful_downloads,
    "failed": len(failed_urls),
    "success_rate": f"{(successful_downloads/len(undownloaded_df)*100):.1f}%" if len(undownloaded_df) > 0 else "N/A"
}

print(f"\nResults:")
print(f"  ✓ Successfully downloaded: {summary['successful']}")
print(f"  ✗ Failed: {summary['failed']}")
print(f"  ✓ Success rate: {summary['success_rate']}")

if failed_urls:
    failed_df = pd.DataFrame({"url": failed_urls, "type": undownloaded_df[undownloaded_df['url'].isin(failed_urls)]['type'].values})
    csv_content = failed_df.to_csv(index=False)
    blob = bucket.blob(FAILED_LINKS_FILE)
    blob.upload_from_string(csv_content, content_type='text/csv')

    failed_df.to_csv("failed_downloads.csv", index=False)
    print(f"\nFailed downloads saved to:")
    print(f"  - GCS: gs://{BUCKET_NAME}/{FAILED_LINKS_FILE}")
    print(f"  - Local: failed_downloads.csv")
else:
    print("\n✓ All downloads completed successfully!")

print("\n" + "="*70)
