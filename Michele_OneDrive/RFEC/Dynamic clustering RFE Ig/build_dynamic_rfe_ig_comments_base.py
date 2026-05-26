#!/usr/bin/env python3
"""
Build the base Instagram comments dataset for rolling four-month RFE analysis.

Current setup:
- keep comments from 2023-01-01 through 2026-03-31
- remove creator comments (Camihawke)
- remove users with only one comment in the selected window
- preserve the original comment columns and append a stable user_key
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_CLUSTER_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = SINGLE_CLUSTER_DIR.parent

COMMENTS_PATH = WORKSPACE_DIR / "ig_comments_clean.csv"
OUTPUT_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_input.csv"

START_DATE = pd.Timestamp("2023-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-04-01")
CREATOR_USERNAME = "camihawke"


def build_user_key(comments: pd.DataFrame) -> pd.Series:
    from_username = comments["from_username"].fillna("").astype("string").str.strip()
    from_username_norm = from_username.str.casefold()
    from_id = comments["from_id"].fillna("").astype("string").str.strip()
    from_id = from_id.mask(from_id == "")
    return from_id.fillna("username:" + from_username_norm)


def load_comments() -> pd.DataFrame:
    comments = pd.read_csv(
        COMMENTS_PATH,
        dtype={
            "comment_id": "string",
            "media_id": "string",
            "timestamp": "string",
            "from_id": "string",
            "from_username": "string",
            "parent_id": "string",
            "is_reply": "boolean",
            "is_creator": "boolean",
        },
    )
    comments["timestamp"] = pd.to_datetime(comments["timestamp"], errors="coerce")
    comments = comments.dropna(subset=["comment_id", "media_id", "timestamp"]).copy()
    comments = comments[(comments["timestamp"] >= START_DATE) & (comments["timestamp"] < END_EXCLUSIVE)].copy()
    return comments


def filter_comments(comments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {}
    stats["rows_in_window"] = len(comments)

    comments["from_username"] = comments["from_username"].fillna("").astype("string").str.strip()
    creator_mask = (comments["is_creator"] == True) | (
        comments["from_username"].str.casefold() == CREATOR_USERNAME
    )
    stats["creator_comments_removed"] = int(creator_mask.sum())
    comments = comments.loc[~creator_mask].copy()
    stats["rows_after_creator_filter"] = len(comments)

    comments["user_key"] = build_user_key(comments)

    user_comment_counts = comments.groupby("user_key").size()
    keep_user_keys = user_comment_counts[user_comment_counts > 1].index
    stats["single_comment_users_removed"] = int((user_comment_counts == 1).sum())

    filtered = comments[comments["user_key"].isin(keep_user_keys)].copy()
    filtered = filtered.sort_values(["timestamp", "media_id", "comment_id"], kind="mergesort").reset_index(
        drop=True
    )

    stats["rows_final"] = len(filtered)
    stats["users_final"] = int(filtered["user_key"].nunique())
    return filtered, stats


def main() -> None:
    comments = load_comments()
    filtered, stats = filter_comments(comments)
    filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {stats['rows_final']:,} rows to {OUTPUT_PATH}")
    print(f"Date window: {START_DATE.date()} to {(END_EXCLUSIVE - pd.Timedelta(days=1)).date()}")
    print(f"Rows in window: {stats['rows_in_window']:,}")
    print(f"Creator comments removed: {stats['creator_comments_removed']:,}")
    print(f"Single-comment users removed: {stats['single_comment_users_removed']:,}")
    print(f"Users retained: {stats['users_final']:,}")


if __name__ == "__main__":
    main()
