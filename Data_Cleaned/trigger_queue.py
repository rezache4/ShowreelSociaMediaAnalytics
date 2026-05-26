import json
import sys
print('EXE:', sys.executable)

import pandas as pd

try:
    from google.cloud import pubsub_v1
except ImportError as e:
    raise ImportError(
        "google-cloud-pubsub is required. Run this script with the showreel environment "
        "or install google-cloud-pubsub into the current interpreter."
    ) from e


# --- CONFIGURATION ---
PROJECT_ID = "project-10b142ae-d53f-4f87-81e" 
TOPIC_ID = "video-fetch-queue"
PARQUET_FILE = r"C:\Users\Mohammad Reza\OneDrive - Politecnico di Milano\Showreel\Data_Cleaned\yt_videos_cleaned_with_transcripts.parquet"  # <-- INSERT YOUR FILE PATH
COLUMN_NAME = "videoId"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

print(f"Loading {PARQUET_FILE}...")
df = pd.read_parquet(PARQUET_FILE)

# Extract unique video IDs, dropping any empty rows
video_ids = df[COLUMN_NAME].dropna().unique()
total_videos = len(video_ids)
print(f"Found {total_videos} videos. Sending to Google Cloud...")

# Loop through and publish each one to the queue
count = 0
for vid in video_ids:
    message_data = json.dumps({"videoId": str(vid)})
    publisher.publish(topic_path, data=message_data.encode("utf-8"))
    
    count += 1
    if count % 1000 == 0:
        print(f"Queued {count}/{total_videos}...")

print("SUCCESS: All videos have been dropped into the queue.")