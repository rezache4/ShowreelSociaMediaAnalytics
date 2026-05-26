"""Merge local transcripts into `yt_videos_cleaned.parquet`.

This script discovers transcripts under `transcripts/` and `transcripts/cloud_transcripts/`,
strips timing markers from VTT/SRT/TXT, and merges cleaned text into the dataframe as
`local_transcript`.

Usage:
    python Data_Cleaned/pipeline_tools/add_transcripts.py --input Data_Cleaned/yt_videos_cleaned.parquet --output Data_Cleaned/yt_videos_with_local_transcripts.parquet

Optional flags:
    --inplace    Overwrite the input parquet file.

Dependencies: pandas, pyarrow
"""
import argparse
import os
import re
from pathlib import Path
import pandas as pd


def strip_timestamps_and_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove WEBVTT header and metadata
    text = re.sub(r"(?i)webvtt(:?.*)?\n", "", text)
    # Remove common timestamp lines (SRT/VTT) and arrows
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"\d{1,2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}[\.,]\d+", "", text)
    # Remove standalone timestamps
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"\d{1,2}:\d{2}[\.,]\d+", "", text)
    # Remove SRT numeric indices
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove html tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove extra whitespace and join lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.match(r"^\s*$", ln)]
    return " ".join(lines).strip()


def discover_transcripts(transcripts_root: str):
    transcripts = {}
    root = Path(transcripts_root)
    if not root.exists():
        return transcripts
    for p in root.rglob('*'):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in ['.vtt', '.srt', '.txt', '.empty'] or p.parent.name == 'cloud_transcripts':
                try:
                    text = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    text = ''
                cleaned = strip_timestamps_and_tags(text)
                # derive video id from filename: remove leading underscores and known language tags
                name = p.name.lstrip('_')
                # remove language/extra suffixes like .it, .en before extension
                name = re.split(r"\.(?=[^\.]+$)", name)[0]
                # sometimes names include dots: take first token
                vid = name.split('.')[0]
                transcripts[vid] = cleaned
    return transcripts


def detect_id_column(df: pd.DataFrame):
    candidates = ['video_id', 'id', 'videoId', 'videoID', 'yt_id', 'youtube_id']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try to find any column that looks like video ids (length ~11)
    for c in df.columns:
        sample = df[c].dropna().astype(str)
        if not sample.empty and sample.iloc[0] and 5 <= len(sample.iloc[0]) <= 50:
            return c
    return None


def main(args):
    df = pd.read_parquet(args.input)
    id_col = detect_id_column(df)
    if id_col is None:
        raise RuntimeError('Could not detect a video id column in the dataframe. Please inspect columns: ' + ','.join(df.columns))

    transcripts_map = discover_transcripts(os.path.join(Path(args.input).parents[0].as_posix().replace('\\', '/'), '..', 'transcripts'))
    # fallback: also search workspace top-level transcripts folder
    if not transcripts_map:
        transcripts_map = discover_transcripts(os.path.join(Path(args.input).parents[1].as_posix().replace('\\', '/'), 'transcripts'))
    # simpler: also try './transcripts'
    if not transcripts_map:
        transcripts_map = discover_transcripts('transcripts')

    local_texts = []

    for _, row in df.iterrows():
        vid = str(row[id_col]).lstrip('_')
        # sometimes video ids are longer with prefixes; accept first token
        vid = vid.split('.')[0]
        # pick local transcript if available
        text = transcripts_map.get(vid, '')
        local_texts.append(text if text else pd.NA)

    df['local_transcript'] = local_texts

    out_path = args.output
    if args.inplace:
        out_path = args.input
    df.to_parquet(out_path, index=False)
    print(f'Wrote updated parquet to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--output', required=False, help='Output parquet path', default='Data_Cleaned/yt_videos_with_local_transcripts.parquet')
    parser.add_argument('--inplace', action='store_true', help='Overwrite input file')
    args = parser.parse_args()
    main(args)
