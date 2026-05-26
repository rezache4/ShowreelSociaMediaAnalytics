# Video Processing Pipeline - Team Setup Guide

## Quick Start

This template notebook allows each team member to process their own video datasets independently without affecting their system Python installation.

### Prerequisites
- **Python 3.8+** installed on your system
- **VS Code** with Jupyter extension (or Jupyter Lab)
- ~5-10 GB disk space for videos + processed data

### System Requirements by OS

#### Windows
- FFmpeg will be auto-installed via Chocolatey (if available)
- Alternative: Manual installation from https://ffmpeg.org/download.html

#### macOS
- FFmpeg auto-installed via Homebrew (`brew install ffmpeg`)
- Requires Homebrew: https://brew.sh

#### Linux
- FFmpeg auto-installed via apt (`sudo apt-get install ffmpeg`)

---

## Setup Steps (First Time Only)

### 1. **Clone/Download the Notebook**
   - Download `Team_Video_Processing_Template.ipynb` from the shared folder
   - Create a new folder for your project (e.g., `my_video_processing/`)
   - Place the notebook in this folder

### 2. **Prepare Your Dataset**
   You need a list of video links. Options:
   
   - **CSV File**: Create a CSV with a column containing URLs
     ```csv
     url
     https://www.instagram.com/p/ABC123/
     https://www.instagram.com/p/XYZ789/
     ```
   
   - **Simple List**: Directly paste URLs in the notebook (STEP 2)
   
   - **Any Other Format**: Load however you prefer (JSON, Excel, etc.)

   ### 2b. **Access Shared Processed Dataset (multimodal_dataset_fixed)**
   The team shares a pre-processed dataset in Google Drive:
   https://drive.google.com/drive/u/0/folders/1_MRp6U-77zVCy8bJfEsvXk8tKUUCRXoE

   **Recommended (Google Drive Desktop sync):**
   1. Install Google Drive for Desktop: https://www.google.com/drive/download/
   2. Add the shared folder to your Drive.
   3. Choose a local sync path (e.g., `D:\Data\AFB_Lab\multimodal_dataset_fixed`).
   4. In your code/notebooks, point `multimodal_dataset_fixed` to that local path.

   **Manual download (if you do not want sync):**
   1. Open the Drive folder link above.
   2. Download the folder to your machine.
   3. Place it at a known local path and update your notebooks to that path.

### 3. **Open the Notebook**
   ```bash
   # If using Jupyter Lab
   jupyter lab Team_Video_Processing_Template.ipynb
   
   # If using VS Code with Jupyter extension
   # Just open the file and click "Select Kernel" → "Python environments"
   ```

---

## Running the Notebook

### ⚠️ Important: Run cells IN ORDER

Each team member should follow these steps:

1. **STEP 1: Environment Setup**
   - Run the first code cell
   - Installs Python packages in a local virtual environment
   - **Does NOT affect your system Python**
   - Takes 2-5 minutes first time
   - Safe to run multiple times

2. **STEP 2: Provide Your Video Links**
   - Replace the placeholder with your actual dataset
   - Can be a list, CSV file, DataFrame column, etc.
   - See cell for examples

3. **STEP 3-5: Helper Functions**
   - Just run these (they load models and define functions)
   - No input needed

4. **STEP 6: Process Videos**
   - Main processing starts here
   - Downloads videos, extracts frames, transcribes audio
   - Takes time depending on number of videos and internet speed

5. **STEP 7: Validate Results**
   - Check which videos processed successfully
   - See what components are available for each video

---

## Output Structure

After running, you'll have:

```
your_project_folder/
├── Team_Video_Processing_Template.ipynb (your notebook)
├── processing_venv/ (local Python environment - don't touch)
├── multimodal_dataset/ (OUTPUT - all processed videos)
│   ├── VIDEO_ID_1/
│   │   ├── VIDEO_ID_1.mp4
│   │   ├── VIDEO_ID_1.info.json (metadata)
│   │   ├── transcription.txt (with timestamps)
│   │   └── frames/
│   │       ├── frame_00_first.jpg
│   │       ├── frame_01_second.jpg
│   │       └── frame_XXXX_scene.jpg (multiple)
│   ├── VIDEO_ID_2/
│   │   └── ...
├── processed_urls.txt (tracks what's already done)
└── TEAM_SETUP_INSTRUCTIONS.md (this file)
```

---

## Customization

### Adjust Scene Detection Sensitivity
In **STEP 4**, change this line:
```python
SCENE_THRESHOLD = 0.4  # 0.0 = very sensitive, 1.0 = not sensitive
```

### Change Whisper Model Size
In **STEP 3**, change this line:
```python
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="float32")
```

Available models (larger = slower but more accurate):
- `"tiny"` - Fastest
- `"base"` - Default
- `"small"` - Better accuracy
- `"medium"` - High accuracy
- `"large"` - Best accuracy (slowest)

### Change Output Folder
In **STEP 5**, modify:
```python
DATASET_DIR = "multimodal_dataset"  # Change this to any folder name
```

If you want to read the shared dataset, set:
```python
DATASET_DIR = "D:/Data/AFB_Lab/multimodal_dataset_fixed"
```

---

## Troubleshooting

### Problem: "ffmpeg: command not found"

**Windows:**
1. Check if Chocolatey is installed: `choco --version`
2. If not, install from https://chocolatey.org/install
3. Rerun the setup cell
4. If still fails, install FFmpeg manually: https://ffmpeg.org/download.html

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

### Problem: "ModuleNotFoundError" or "No module named..."
- This shouldn't happen if you ran STEP 1 successfully
- Try restarting the Jupyter kernel and running STEP 1 again

### Problem: Videos not downloading
- Some videos may be:
  - Private or age-restricted
  - Region-locked
  - Deleted or unavailable
- Check the VALIDATION REPORT (STEP 7) to see which ones failed

### Problem: Very slow processing
- Transcription (STEP 3) is the slowest part - this is normal
- Each video takes 1-10 minutes depending on length and language
- Patience is required!

### Problem: Running out of disk space
- Check how much space videos need: size varies (typically 10-100 MB each)
- Delete old processed videos from `multimodal_dataset/` if needed
- If you are using the shared `multimodal_dataset_fixed`, store it on a drive with enough space

---

## Resuming Work

### If interrupted mid-processing:
1. Simply run STEP 6 again
2. The notebook automatically skips already-processed URLs (tracked in `processed_urls.txt`)
3. Can be run multiple times safely

### Reusing the same folder:
- Just update the video links in STEP 2
- Keep running STEP 6 to process new URLs
- Old results remain in `multimodal_dataset/`

---

## FAQ

**Q: Will this affect my system Python?**
A: No! Everything runs in a local virtual environment (`processing_venv/`) created by the notebook.

**Q: Can I delete the `processing_venv/` folder?**
A: Yes, but then you'll need to run STEP 1 again next time. It's harmless to keep it.

**Q: Can I run this on multiple machines?**
A: Yes! Each person can have their own copy of the notebook in their own folder. They won't interfere with each other.

**Q: What if two people process the same video?**
A: That's fine! They'll just create separate folders with the same video_id. You can share the results or keep them separate.

**Q: Can I modify the notebook?**
A: Yes! Feel free to customize it. Just don't break the cell order (STEP 1 → STEP 2 → ... → STEP 7).

**Q: Where can I find the downloaded videos?**
A: In the `multimodal_dataset/` folder in your project directory.

---

## Contact

If you run into issues:
1. Check this guide first
2. Check the **Notes & Troubleshooting** section in the notebook itself
3. Reach out to the main team lead

---

## Technical Details

### Why a Virtual Environment?
- Isolates dependencies (yt-dlp, pandas, faster-whisper, etc.)
- Prevents conflicts with your system Python
- Each person can have different versions if needed
- Easy to delete and start fresh (just delete `processing_venv/`)

### How It Works
1. **Cell 1**: Creates `processing_venv/` and installs packages there
2. **Cell 2**: You provide video links
3. **Cells 3-5**: Load the transcription AI model and define helper functions
4. **Cell 6**: Downloads videos using yt-dlp, then:
   - Uses FFmpeg to extract frames
   - Uses Whisper AI to transcribe audio
   - Saves everything organized in `multimodal_dataset/`
5. **Cell 7**: Validates what was successfully processed

### Dependencies Installed
- **yt-dlp**: Downloads videos (Instagram, YouTube, TikTok, etc.)
- **pandas**: Data manipulation
- **faster-whisper**: AI transcription
- **ffmpeg-python**: Python FFmpeg wrapper (plus FFmpeg system library)

---

## Next Steps After Processing

Once you have processed videos in `multimodal_dataset/`, you can:
- Analyze transcriptions
- Use frames for computer vision tasks
- Build datasets with combined data
- Share results with team
- Feed into machine learning models

All videos are organized with consistent structure, making it easy to automate downstream tasks.

---

**Happy processing! 🎥**
