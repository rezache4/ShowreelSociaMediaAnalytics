# Google Colab Enterprise Setup Guide

This guide explains how to run the Instagram download pipeline on Google Colab Enterprise with Google Cloud Storage.

## Overview

There are 2 cells:
1. **Cell 1**: Extract undownloaded image/carousel links
2. **Cell 2**: Download pipeline that saves directly to GCS bucket

## Prerequisites

- Google Colab Enterprise access
- Google Cloud Storage bucket with your dataset
- Instagram credentials (username & password)
- GCS bucket permissions

## Setup Steps

### Step 1: Prepare Your Google Cloud Bucket

Your bucket should have this structure:
```
gs://your-bucket-name/
├── Data/Camihawke/Cleaned/
│   └── ig_posts_with_duration.parquet
└── multimodal_dataset_fixed/
    ├── image/          (existing downloaded images)
    └── carousel/       (existing downloaded carousels)
```

### Step 2: Set Instagram Credentials in Colab

Choose ONE of these methods:

#### Method A: Colab Secrets (Recommended - Most Secure)
1. In Colab, click the "🔑 Secrets" icon in the left sidebar
2. Create two secrets:
   - Name: `IG_USERNAME`, Value: your Instagram username
   - Name: `IG_PASSWORD`, Value: your Instagram password
3. Grant notebook access to both secrets

#### Method B: Environment Variables
Add this to the first cell before running:
```python
import os
os.environ["IG_USERNAME"] = "your_instagram_username"
os.environ["IG_PASSWORD"] = "your_instagram_password"
```

#### Method C: .env File
1. Create `.env` file with:
   ```
   IG_USERNAME=your_instagram_username
   IG_PASSWORD=your_instagram_password
   ```
2. Upload to Colab Files
3. Both cells will automatically load it

### Step 3: Configure the Cells

In **both cells**, update the configuration section:

```python
# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================
BUCKET_NAME = "your-bucket-name"  # ← Change this
GCS_PROJECT_ID = "your-project-id"  # ← Change this
```

Find your bucket name and project ID:
- Bucket name: visible in GCS console (e.g., `my-data-bucket`)
- Project ID: in Google Cloud Console (Settings → Project ID)

### Step 4: Run the Cells

#### Cell 1: Extract Undownloaded Links
```
Copy colab_cell_1_extract_links.py content into a Colab cell
Run it
Output: Creates undownloaded_links.csv in bucket
```

This cell:
- Loads your ig_posts from the bucket
- Checks which images/carousels already exist
- Extracts links for items NOT yet downloaded
- Saves to `undownloaded_links.csv` in the bucket

**Expected output:**
```
✓ Loaded 5000 posts
✓ Found 3000 IMAGE posts
✓ Found 2000 CAROUSEL_ALBUM posts
✓ Already downloaded images: 2500
✓ Already downloaded carousels: 1800
✓ Undownloaded images: 500
✓ Undownloaded carousels: 200
✓ Total to download: 700
```

#### Cell 2: Download Pipeline
```
Copy colab_cell_2_download_pipeline.py content into a Colab cell
Run it
Output: Downloads to bucket + failed_downloads.csv
```

This cell:
- Loads the undownloaded links from Cell 1
- Authenticates with Instagram
- Downloads each image/carousel
- Uploads directly to GCS bucket
- Saves failed links to `failed_downloads.csv`

**Expected output:**
```
✓ Loaded 700 links to download
[1/700] ⬇️ Downloading IMAGE: abc123xyz...
  ⬆️ Uploading to GCS...
  ✓ Success!
  ⏸ Pausing 4.23s (human-like delay)

[2/700] ⬇️ Downloading CAROUSEL_ALBUM: def456uvw...
...
```

## Features

### Human-like Delays
- Pauses between 3-12 seconds between downloads
- Includes Gaussian jitter for variation
- 10% chance of longer pause (simulates thinking)
- Avoids detection by Instagram rate-limiting

### Error Handling
- Auto-retries with re-authentication on login expiry
- Skips items already in bucket (safe for re-runs)
- Saves failed downloads for retry later

### GCS Integration
- Uploads directly to cloud (no local storage needed)
- Efficient streaming upload
- Supports resuming from failures

## Troubleshooting

### "IG_USERNAME not found"
- Ensure you set credentials via Secrets, environment variables, or .env file
- Verify variable names are exactly: `IG_USERNAME` and `IG_PASSWORD`

### "Bucket not found" or authentication errors
- Verify `BUCKET_NAME` and `GCS_PROJECT_ID` are correct
- Check GCS permissions: your Colab account needs access to the bucket
- Run `auth.authenticate_user()` first

### "ig_posts not found in bucket"
- Verify `IG_POSTS_PATH` points to correct location
- Default: `Data/Camihawke/Cleaned/ig_posts_with_duration.parquet`
- Adjust if your path is different

### Downloads stop or timeout
- Increase pause times in `human_like_pause()` min/max seconds
- Check Instagram isn't blocking your account
- Can re-run Cell 2 to resume from where it stopped

### Out of memory in Colab
- This is unlikely since downloads go directly to GCS
- If it happens, restart runtime and re-run cells

## Performance

- Download speed depends on Instagram rate limits
- Each download typically takes 3-15 seconds (including pause)
- 1000 downloads = ~1-3 hours depending on pauses
- GCS upload is fast (< 1 sec per item)

## Safety Notes

- **Never commit credentials to git** - use Secrets instead
- Instagram may temporarily limit downloads if too aggressive
- Always use human-like pauses to avoid detection
- Monitor your Instagram account for login notifications
- Can be interrupted anytime with no data loss (downloads are atomic)

## Next Steps

After downloading:
1. Re-run Cell 1 to check for newly downloaded items
2. Cell 2 will skip already-uploaded files (idempotent)
3. Use failed_downloads.csv to retry specific items
4. Merge downloaded media back into your dataset pipeline

## Need Help?

- Check cell output for specific error messages
- Verify bucket permissions in GCS console
- Test Instagram login with manual browser attempt
- Check Colab runtime logs for API errors
