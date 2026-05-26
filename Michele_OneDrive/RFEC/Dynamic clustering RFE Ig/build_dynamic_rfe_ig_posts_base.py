#!/usr/bin/env python3
"""
Build the base Instagram posts dataset for rolling four-month RFE analysis.

Current setup:
- keep posts from 2023-01-01 through 2026-03-31
- keep only posts whose media_id appears at least once in ig_comments_clean.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_CLUSTER_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SINGLE_CLUSTER_DIR.parent

POSTS_PATH = WORKSPACE_DIR / "ig_posts_clean.csv"
COMMENTS_PATH = WORKSPACE_DIR / "ig_comments_clean.csv"
OUTPUT_PATH = SCRIPT_DIR / "ig_posts_dynamic_rfe_available_comments.csv"

START_DATE = pd.Timestamp("2023-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-04-01")


def load_posts() -> pd.DataFrame:
    posts = pd.read_csv(
        POSTS_PATH,
        dtype={
            "media_id": "string",
            "timestamp": "string",
        },
    )
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")
    posts = posts.dropna(subset=["media_id", "timestamp"]).copy()
    posts = posts[(posts["timestamp"] >= START_DATE) & (posts["timestamp"] < END_EXCLUSIVE)].copy()
    return posts


def load_commented_media_ids() -> pd.Index:
    comments = pd.read_csv(
        COMMENTS_PATH,
        usecols=["media_id"],
        dtype={"media_id": "string"},
    )
    comments = comments.dropna(subset=["media_id"]).copy()
    comments["media_id"] = comments["media_id"].astype("string").str.strip()
    comments = comments[comments["media_id"] != ""].copy()
    return pd.Index(comments["media_id"].drop_duplicates())


def main() -> None:
    posts = load_posts()
    commented_media_ids = load_commented_media_ids()

    input_posts = len(posts)
    filtered = posts[posts["media_id"].isin(commented_media_ids)].copy()
    filtered = filtered.sort_values(["timestamp", "media_id"], kind="mergesort").reset_index(drop=True)

    filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(filtered):,} rows to {OUTPUT_PATH}")
    print(f"Date window: {START_DATE.date()} to {(END_EXCLUSIVE - pd.Timedelta(days=1)).date()}")
    print(f"Input posts in window: {input_posts:,}")
    print(f"Posts removed for missing comments in ig_comments_clean: {input_posts - len(filtered):,}")


if __name__ == "__main__":
    main()
