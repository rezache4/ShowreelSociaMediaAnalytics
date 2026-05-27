# Quick Start: 5 Steps

## Step 1: LOCAL - Extract Downloaded Links (2 min)

In your **IG_Download.ipynb** notebook:
1. Create new cell
2. Copy code from: `notebook_cell_extract_downloaded_links.py`
3. Run it
4. **Output:** `downloaded_links.csv` (in your working directory)

```python
# Command to check you have the file:
import os
print(os.path.exists("downloaded_links.csv"))  # Should print: True
```

---

## Step 2: COLAB - Upload Credentials

1. Open Google Colab: https://colab.research.google.com
2. Create new notebook
3. In left sidebar, click **🔑 Secrets** icon
4. Create two secrets:
   - `IG_USERNAME` = your Instagram username
   - `IG_PASSWORD` = your Instagram password
5. Click "Grant notebook access" for each

---

## Step 3: COLAB - Cell 1 (Extract Undownloaded)

In Colab notebook:

1. Create new cell
2. Copy code from: `colab_cell_1_extract_links.py`
3. Update at top:
   ```python
   BUCKET_NAME = "your-actual-bucket-name"
   GCS_PROJECT_ID = "your-actual-project-id"
   ```
4. Run cell ⏱️ ~2-5 minutes

**Output:** `undownloaded_links.csv` created in bucket

---

## Step 4: COLAB - Cell 2 (Download Everything)

In same Colab notebook:

1. Create new cell
2. Copy code from: `colab_cell_2_download_pipeline.py`
3. Update same config:
   ```python
   BUCKET_NAME = "your-actual-bucket-name"
   GCS_PROJECT_ID = "your-actual-project-id"
   ```
4. Run cell ⏱️ Duration = 5-10 min per 100 items

**Monitoring:**
- Watch progress bar
- Each item shows ✓ or ✗
- Pauses happen automatically

**Output:** `failed_downloads.csv` (if any failed)

---

## Step 5: COLAB - Cell 3 (Optional - Verify)

To double-check everything uploaded:

1. Create new cell
2. Copy code from: `colab_cell_verify_downloaded_links.py`
3. Update same config
4. Run cell ⏱️ ~1-2 minutes

**Output:** Verification report + `missing_from_bucket.csv`

---

## 🎯 That's It!

Your images and carousels are now in:
- `gs://your-bucket-name/multimodal_dataset_fixed/image/`
- `gs://your-bucket-name/multimodal_dataset_fixed/carousel/`

---

## 🔥 If Something Goes Wrong

| Problem | Solution |
|---------|----------|
| "Bucket not found" | Check `BUCKET_NAME` spelling |
| "Permission denied" | Verify GCS bucket permissions |
| "IG_USERNAME not found" | Create Secrets in Colab (Step 2) |
| Downloads stopped | Re-run Cell 2 (it resumes) |
| Network error | Restart cell and try again |

---

## 📊 Expected Performance

- **Cell 1:** 2-5 minutes (depends on dataset size)
- **Cell 2:** 3-6 hours for 1000 items (includes pauses to avoid detection)
- **Cell 3:** 1-2 minutes

Total time for 1000 items: **4-6 hours** (mostly waiting for downloads)

---

## 💾 Files You Need

1. ✅ `notebook_cell_extract_downloaded_links.py` — Run locally
2. ✅ `colab_cell_1_extract_links.py` — Run on Colab
3. ✅ `colab_cell_2_download_pipeline.py` — Run on Colab
4. ✅ `colab_cell_verify_downloaded_links.py` — Optional, run on Colab

---

## 📝 Configuration Template

Save this and update:

```python
# Configuration to use in Colab Cells 1, 2, and 3
BUCKET_NAME = "my-data-bucket"
GCS_PROJECT_ID = "my-project-id-12345"
```

Find these values:
- **BUCKET_NAME:** In GCS console → Buckets → your bucket name
- **GCS_PROJECT_ID:** In Google Cloud Console → Settings → Project ID

---

## ✨ That's it! Questions?

- Check `COLAB_WORKFLOW.md` for detailed explanation
- Check `COLAB_SETUP_GUIDE.md` for troubleshooting
