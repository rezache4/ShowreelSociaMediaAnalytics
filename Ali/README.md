# Ali - Notebook Work Summary

This folder contains my working notebooks, datasets, and outputs for the Instagram video download and processing pipeline. The goal is to build a clean multimodal dataset (video, frames, transcripts, metadata) and track download completeness.

## Contents

- Notebooks: analysis, download, and processing workflows.
- Data/: cleaned datasets used as inputs (Camihawke, YouTube, etc.).
- multimodal_dataset/: raw processed outputs (one folder per video).
- multimodal_dataset_fixed/: reorganized outputs split into feed/ and reel/.
- feed_links.txt, reels_links.txt, processed_urls.txt: URL lists and progress tracking.
- yt_videos_cleaned_with_transcripts.parquet: auxiliary dataset.

## What the main notebook does (IG_Download.ipynb)

### 1) Environment setup
- Creates a local virtual environment (processing_venv).
- Installs required packages: yt-dlp, pandas, faster-whisper, ffmpeg-python, google-cloud-storage.
- Ensures FFmpeg is installed (Chocolatey on Windows, apt on Linux, brew on macOS).

### 2) Load cleaned Instagram datasets
- Loads Camihawke cleaned parquet files (ig_posts, ig_comments) from Data/.
- Uses ig_posts to derive video links (permalink).

### 3) Build link lists
- Splits Instagram links by media type (VIDEO, CAROUSEL, IMAGE).
- Further splits VIDEO posts into FEED vs REELS (media_product_type).
- Writes feed_links.txt and reels_links.txt.

### 4) Download and process videos
- Checks what is already downloaded by matching URLs to existing folders or .info.json metadata.
- Downloads missing URLs with yt-dlp using IG credentials from .env or .env.example.
- Writes .info.json metadata to each video folder.
- Extracts frames (first frame, 1-second frame, and scene-change frames).
- Transcribes audio with faster-whisper to transcription.txt.
- Logs successes and failures (failed_downloads.csv), and appends to processed_urls.txt.

### 5) Reorganize into multimodal_dataset_fixed
- Copies each video folder into feed/ or reel/ based on URL matching.
- Skips unknown/unmatched items.

### 6) Quality and completeness checks
- Verifies for each URL whether info.json, video file, frames, and transcript exist.
- Generates video_lengths_seconds.csv with durations via ffprobe.
- Plots a distribution of video durations.

## Expected folder structure

multimodal_dataset/
  VIDEO_ID/
    VIDEO_ID.mp4
    VIDEO_ID.info.json
    transcription.txt
    frames/
      frame_00_first.jpg
      frame_01_second.jpg
      frame_XXXX_scene.jpg

multimodal_dataset_fixed/
  feed/
    VIDEO_ID/
  reel/
    VIDEO_ID/

## Credentials
- The notebook loads IG credentials from:
  - Ali/.env or Ali/.env.example
  - ../.env or ../.env.example (repo root)
- Required variables: IG_USERNAME, IG_PASSWORD.

## How to extend or reuse

- Add new URLs to the input datasets or directly in the notebook.
- Re-run the download step; existing downloads are skipped by URL match.
- Use the completeness checker to identify partial downloads and re-run failed URLs.
- If the dataset moves, update DATASET_DIR in relevant cells.

## Notes for future agent work

- The core logic lives in IG_Download.ipynb.
- Avoid running multiple download loops in parallel to prevent throttling or credential lockouts.
- Path assumptions are relative to Ali/ (since notebooks and data are inside this folder).
- multimodal_dataset and multimodal_dataset_fixed are intentionally ignored in git.
