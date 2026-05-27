# Complete Workflow: Local → Colab → GCS

This document explains the complete workflow for extracting, uploading, and continuing downloads on Colab Enterprise.

## Overview

```
┌─────────────────────────────────────┐
│  1. LOCAL NOTEBOOK                  │
│  Extract downloaded_links.csv       │
│  (What's already been downloaded)   │
└────────────┬────────────────────────┘
             │ Download file
             ▼
┌─────────────────────────────────────┐
│  2. UPLOAD TO COLAB FILES          │
│  Upload downloaded_links.csv        │
│  (Import to Colab environment)      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. COLAB CELL 1                   │
│  Extract undownloaded links         │
│  (What's missing from bucket)       │
│  → undownloaded_links.csv           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. COLAB CELL 2                   │
│  Download pipeline                  │
│  Downloads & uploads to bucket      │
│  → failed_downloads.csv             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. COLAB CELL 3 (Optional)        │
│  Verify all items in bucket         │
│  → verification report              │
└─────────────────────────────────────┘
```

## Step-by-Step Instructions

### Step 1: Extract Downloaded Links (Local Notebook)

**In your local Jupyter notebook (IG_Download.ipynb):**

1. Create a new cell at the end
2. Copy the code from `notebook_cell_extract_downloaded_links.py`
3. Run the cell

**Output:**
- `downloaded_links.csv` — Full list with shortcodes and types
- `downloaded_links_urls_only.csv` — Simplified version (URLs only)

**Example output:**
```
✓ Found 2500 downloaded IMAGE items
✓ Found 1800 downloaded CAROUSEL_ALBUM items
✓ Total downloaded items: 4300

✓ Saved to local file: downloaded_links.csv
✓ Saved simplified version: downloaded_links_urls_only.csv
```

### Step 2: Upload to Colab

**In Google Colab:**

1. Click the **Files** folder icon in left sidebar
2. Click **Upload** button
3. Select `downloaded_links.csv` from your local directory
4. Wait for upload to complete (shows file in Files panel)

**Alternative (using direct upload):**
```python
from google.colab import files
files.upload()  # Opens file picker
```

### Step 3: Colab Cell 1 - Extract Undownloaded Links

**Create a new cell in Colab:**

1. Copy code from `colab_cell_1_extract_links.py`
2. **Verify you uploaded `downloaded_links.csv`** to Colab Files
3. **Update configuration:**
   ```python
   BUCKET_NAME = "your-bucket-name"
   GCS_PROJECT_ID = "your-project-id"
   ```
4. Run the cell

**What it does:**
- Loads your `downloaded_links.csv` (from local notebook)
- Loads `ig_posts_with_duration.parquet` from bucket
- Compares them to find what's NOT yet downloaded
- Saves `undownloaded_links.csv` to bucket

**Output:**
```
✓ Loaded 4300 already downloaded items
  • IMAGE: 2500
  • CAROUSEL_ALBUM: 1800
✓ Loaded 5000 total posts
  • IMAGE: 3000
  • CAROUSEL_ALBUM: 2000
✓ Undownloaded items found:
  • IMAGE: 500
  • CAROUSEL_ALBUM: 200
  • Total to download: 700
```

### Step 4: Colab Cell 2 - Download Pipeline

**In the same Colab notebook:**

1. Create a new cell
2. Copy code from `colab_cell_2_download_pipeline.py`
3. **Update configuration** (same as Cell 1):
   ```python
   BUCKET_NAME = "your-bucket-name"
   GCS_PROJECT_ID = "your-project-id"
   ```
4. **Ensure credentials are set** via:
   - Colab Secrets (recommended): `IG_USERNAME`, `IG_PASSWORD`
   - Environment variables
   - Or uploaded `.env` file
5. Run the cell

**What it does:**
- Loads undownloaded links from bucket
- Authenticates with Instagram
- Downloads each image/carousel
- Uploads to bucket
- Saves failed items to `failed_downloads.csv`

**Monitoring:**
- Check download progress in real-time
- Each item shows status: ✓ Success or ✗ Failed
- Pause times vary (3-15 seconds) to avoid detection

**Expected duration:**
- 100 items: ~20-30 minutes
- 500 items: ~2-3 hours
- 1000 items: ~4-6 hours
- (Depends on pause times and Instagram rate limits)

### Step 5 (Optional): Colab Cell 3 - Verify Downloads

**To verify everything uploaded correctly:**

1. Create a new cell
2. Copy code from `colab_cell_verify_downloaded_links.py`
3. Update configuration (same as Cells 1 & 2)
4. Run the cell

**What it does:**
- Verifies each item from `downloaded_links.csv` exists in bucket
- Generates verification report
- Saves any missing items to `missing_from_bucket.csv`

**Output:**
```
Verification Report:
  • total_items: 4300
  • verified: 4298
  • missing: 2
  • verification_rate: 99.95%

Missing items saved to: missing_from_bucket.csv
```

## File Reference

| File | Location | Purpose |
|------|----------|---------|
| `downloaded_links.csv` | Local → Upload to Colab | List of what you've downloaded |
| `downloaded_links_urls_only.csv` | Local (optional) | Simplified version |
| `undownloaded_links.csv` | Bucket | Generated by Cell 1 |
| `failed_downloads.csv` | Bucket | Generated by Cell 2 |
| `missing_from_bucket.csv` | Bucket (optional) | Generated by Cell 3 |

## Troubleshooting

### "downloaded_links.csv not found in Colab"
- Verify you uploaded the file to Colab Files
- Check spelling matches exactly
- Try re-uploading the file

### Cell 1 shows "Already downloaded: 0"
- This is normal! It means Cell 1 doesn't use your local CSV
- Cell 1 checks the actual bucket to see what's there
- It will find your previously uploaded items

### Cell 2 gets stuck or times out
- Check your internet connection
- Instagram may be rate-limiting
- You can re-run Cell 2 to resume (it skips done items)

### Downloads are very slow
- This is normal - the human-like pauses add time
- Increase `max_seconds` in `human_like_pause()` if you want faster downloads
- But slower = lower detection risk

### "Failed to authenticate"
- Verify Instagram username/password are correct
- Check for special characters that need escaping
- Try logging in manually to Instagram (check for security alerts)

## Best Practices

✅ **Do:**
- Run Cell 1 first to check what needs downloading
- Monitor Cell 2 output for errors
- Use Colab Secrets for credentials (most secure)
- Let downloads complete fully before closing browser

✗ **Don't:**
- Interrupt Cell 2 mid-download (items are atomic)
- Close Colab tab while downloading (use Colab background execution)
- Run Cell 2 multiple times in parallel (use same session)
- Commit credentials to git/notebooks

## Resuming Failed Downloads

If Cell 2 fails partway through:

1. Check `failed_downloads.csv` in your bucket
2. Create a new `retry_links.csv` with the failed items
3. Modify Cell 1 to use `retry_links.csv` instead
4. Re-run Cell 2

## Next Steps After Downloading

Once all downloads complete:

1. Run Cell 3 to verify everything is in the bucket
2. Update your local `ig_posts` with the new downloaded items
3. Merge the bucket data back into your main dataset
4. Run analysis on the complete dataset

## Questions?

Refer to `COLAB_SETUP_GUIDE.md` for detailed setup instructions.
