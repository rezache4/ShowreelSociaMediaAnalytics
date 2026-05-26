#!/usr/bin/env python3
"""
Build era-level user RFE metrics for Instagram comments.

Cleaning choices applied before the metrics:
- keep only non-creator comments
- use `from_id` when available and fall back to a username-based key otherwise

Metrics per user within each era:
- recency: posts published after the user's last comment, divided by total era posts
- frequency: number of distinct posts commented by the user within the era
- engagement: average words per comment, counting any contiguous emoji run as 1 word
- clumpiness: population std. dev. of the number of posts published between
  consecutive comments by the same user inside the era
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent


COMMENT_COLUMNS = [
    "comment_id",
    "media_id",
    "text",
    "timestamp",
    "from_id",
    "from_username",
    "is_reply",
    "is_creator",
]

COMMENT_DTYPES = {
    "comment_id": "string",
    "media_id": "string",
    "text": "string",
    "timestamp": "string",
    "from_id": "string",
    "from_username": "string",
    "is_reply": "boolean",
    "is_creator": "boolean",
}


TARGET_COLUMNS = [
    "era",
    "era_start",
    "era_end",
    "era_total_days",
    "posts_in_era",
    "user_key",
    "user_id",
    "username",
    "comment_count",
    "distinct_posts_count",
    "last_comment_timestamp",
    "recency_posts_since_last_comment",
    "recency",
    "frequency",
    "engagement",
    "clumpiness",
]

NON_EMOJI_TOKEN_RE = re.compile(
    r"https?://\S+|[#@]?\w+(?:[.'’_-]\w+)*",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class EraSpec:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def end_exclusive(self) -> pd.Timestamp:
        return self.end + pd.Timedelta(days=1)

    @property
    def total_days(self) -> int:
        return int((self.end - self.start).days + 1)


ERA_SPECS = [
    EraSpec("2017-2019", pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")),
    EraSpec("2020-2022", pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    EraSpec("2023-2026", pd.Timestamp("2023-01-01"), pd.Timestamp("2026-03-31")),
]

ERA_ORDER = [era.label for era in ERA_SPECS]
GLOBAL_START = ERA_SPECS[0].start
GLOBAL_END_EXCLUSIVE = ERA_SPECS[-1].end_exclusive

EMOJI_RANGES = (
    # These ranges cover the standard emoji blocks used here, including
    # skulls, heart variants/colors, and hand-gesture emoji.
    (0x00A9, 0x00A9),
    (0x00AE, 0x00AE),
    (0x203C, 0x203C),
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x2199),
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x23CF, 0x23CF),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1FAFF),
)

EMOJI_CONNECTORS = {
    0x200D,  # zero-width joiner
    0x20E3,  # keycap enclosing mark
    0xFE0E,  # text presentation selector
    0xFE0F,  # emoji presentation selector
}

KEYCAP_BASE_CHARS = set("0123456789#*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ig_comments_RFE_macro.csv from cleaned Instagram comments."
    )
    parser.add_argument(
        "--comments",
        default=WORKSPACE_DIR / "ig_comments_clean.csv",
        type=Path,
        help="Path to the Instagram comments CSV.",
    )
    parser.add_argument(
        "--posts",
        default=WORKSPACE_DIR / "ig_posts_clean.csv",
        type=Path,
        help="Path to the Instagram posts CSV.",
    )
    parser.add_argument(
        "--output",
        default=SCRIPT_DIR / "ig_comments_RFE_macro.csv",
        type=Path,
        help="Output macro CSV path.",
    )
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="Keep only top-level comments (`is_reply != True`).",
    )
    return parser.parse_args()


def is_emojiish_char(char: str) -> bool:
    codepoint = ord(char)
    if codepoint in EMOJI_CONNECTORS or 0x1F3FB <= codepoint <= 0x1F3FF:
        return True
    for start, end in EMOJI_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def consume_keycap_sequence(value: str, start: int) -> int | None:
    if value[start] not in KEYCAP_BASE_CHARS:
        return None

    end = start + 1
    if end < len(value) and ord(value[end]) == 0xFE0F:
        end += 1

    if end < len(value) and ord(value[end]) == 0x20E3:
        return end + 1

    return None


def count_words_with_emoji(text: object) -> int:
    if pd.isna(text):
        return 0

    value = str(text).strip()
    if not value:
        return 0

    count = 0
    in_emoji_run = False
    non_emoji_chunk: list[str] = []

    def flush_non_emoji() -> int:
        nonlocal non_emoji_chunk
        if not non_emoji_chunk:
            return 0
        piece = "".join(non_emoji_chunk)
        non_emoji_chunk = []
        return len(NON_EMOJI_TOKEN_RE.findall(piece))

    index = 0
    while index < len(value):
        char = value[index]

        if char.isspace():
            count += flush_non_emoji()
            in_emoji_run = False
            index += 1
            continue

        keycap_end = consume_keycap_sequence(value, index)
        if keycap_end is not None:
            count += flush_non_emoji()
            if not in_emoji_run:
                count += 1
                in_emoji_run = True
            index = keycap_end
            continue

        if is_emojiish_char(char):
            count += flush_non_emoji()
            if not in_emoji_run:
                count += 1
                in_emoji_run = True
            index += 1
            continue

        in_emoji_run = False
        non_emoji_chunk.append(char)
        index += 1

    count += flush_non_emoji()
    return count


def assign_era(timestamps: pd.Series) -> pd.Categorical:
    eras = pd.Series(pd.NA, index=timestamps.index, dtype="object")
    for era in ERA_SPECS:
        mask = (timestamps >= era.start) & (timestamps < era.end_exclusive)
        eras.loc[mask] = era.label
    return pd.Categorical(eras, categories=ERA_ORDER, ordered=True)


def load_posts(path: Path) -> pd.DataFrame:
    posts = pd.read_csv(
        path,
        usecols=["media_id", "timestamp"],
        dtype={"media_id": "string", "timestamp": "string"},
    )
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")
    posts = posts.dropna(subset=["media_id", "timestamp"]).copy()
    posts = posts[(posts["timestamp"] >= GLOBAL_START) & (posts["timestamp"] < GLOBAL_END_EXCLUSIVE)]
    posts["era"] = assign_era(posts["timestamp"])
    posts = posts.dropna(subset=["era"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return posts


def read_raw_comments(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=COMMENT_COLUMNS, dtype=COMMENT_DTYPES)


def clean_comments(comments: pd.DataFrame) -> pd.DataFrame:
    comments = comments.copy()
    comments["timestamp"] = pd.to_datetime(comments["timestamp"], errors="coerce")
    comments = comments.dropna(subset=["media_id", "timestamp", "from_username"]).copy()
    comments = comments[(comments["timestamp"] >= GLOBAL_START) & (comments["timestamp"] < GLOBAL_END_EXCLUSIVE)]
    comments = comments[comments["is_creator"] != True].copy()

    comments["text"] = comments["text"].fillna("").astype("string")
    comments["from_username"] = comments["from_username"].str.strip()
    comments["from_username_norm"] = comments["from_username"].str.casefold()
    comments["from_id"] = comments["from_id"].str.strip()
    comments.loc[comments["from_id"] == "", "from_id"] = pd.NA

    comments["user_key"] = comments["from_id"].fillna("username:" + comments["from_username_norm"])
    comments["era"] = assign_era(comments["timestamp"])
    comments = comments.dropna(subset=["era"]).copy()
    comments["word_count"] = comments["text"].map(count_words_with_emoji).astype(int)
    return comments


def keep_top_level_comments(comments: pd.DataFrame) -> pd.DataFrame:
    top_level_mask = comments["is_reply"] != True
    return comments.loc[top_level_mask].copy()


def build_era_metrics(era: EraSpec, comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    era_comments = comments.loc[comments["era"] == era.label].copy()
    era_posts = posts.loc[posts["era"] == era.label].copy()
    posts_in_era = int(len(era_posts))

    if posts_in_era == 0:
        raise ValueError(f"No posts found for era {era.label}.")

    user_counts = era_comments.groupby("user_key").size().rename("comment_count")
    eligible_users = user_counts[user_counts >= 3]
    if eligible_users.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    era_comments = era_comments[era_comments["user_key"].isin(eligible_users.index)].copy()
    era_comments = era_comments.sort_values(
        ["user_key", "timestamp", "comment_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    posts_seen = np.searchsorted(
        era_posts["timestamp"].to_numpy(dtype="datetime64[ns]"),
        era_comments["timestamp"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    era_comments["posts_seen_by_comment_time"] = posts_seen
    era_comments["post_gap"] = era_comments.groupby("user_key")["posts_seen_by_comment_time"].diff()

    aggregated = (
        era_comments.groupby("user_key", sort=False)
        .agg(
            user_id=("from_id", "last"),
            username=("from_username", "last"),
            comment_count=("comment_id", "size"),
            distinct_posts_count=("media_id", "nunique"),
            last_comment_timestamp=("timestamp", "max"),
            last_posts_seen=("posts_seen_by_comment_time", "last"),
            engagement=("word_count", "mean"),
            clumpiness=("post_gap", lambda series: float(series.dropna().std(ddof=0))),
        )
        .reset_index()
    )

    aggregated["recency_posts_since_last_comment"] = (
        posts_in_era - aggregated["last_posts_seen"]
    ).astype(int)
    aggregated["recency"] = (
        aggregated["recency_posts_since_last_comment"] / posts_in_era
    )
    aggregated["frequency"] = pd.to_numeric(
        aggregated["distinct_posts_count"],
        errors="coerce",
    )
    aggregated["era"] = era.label
    aggregated["era_start"] = era.start.date().isoformat()
    aggregated["era_end"] = era.end.date().isoformat()
    aggregated["era_total_days"] = era.total_days
    aggregated["posts_in_era"] = posts_in_era

    return aggregated[TARGET_COLUMNS]


def build_macro(
    comments_path: Path,
    posts_path: Path,
    top_level_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    comments_raw = read_raw_comments(comments_path)
    stats = {
        "input_comments": int(len(comments_raw)),
        "creator_comments_removed": int((comments_raw["is_creator"] == True).sum()),
        "reply_comments_input": int((comments_raw["is_reply"] == True).sum()),
    }

    posts = load_posts(posts_path)
    comments = clean_comments(comments_raw)
    if top_level_only:
        stats["reply_comments_removed"] = int((comments["is_reply"] == True).sum())
        comments = keep_top_level_comments(comments)
    else:
        stats["reply_comments_removed"] = 0
    stats["comments_after_cleaning"] = int(len(comments))

    results = [build_era_metrics(era, comments, posts) for era in ERA_SPECS]
    macro = pd.concat(results, ignore_index=True)
    macro["era"] = pd.Categorical(macro["era"], categories=ERA_ORDER, ordered=True)
    macro = macro.sort_values(["era", "user_key"], kind="mergesort").reset_index(drop=True)
    macro["era"] = macro["era"].astype("string")
    stats["output_rows"] = int(len(macro))
    return macro, stats


def main() -> None:
    args = parse_args()
    macro, stats = build_macro(args.comments, args.posts, top_level_only=args.top_level_only)
    macro.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved {len(macro):,} rows to {args.output}")
    print(f"Input comments: {stats['input_comments']:,}")
    print(f"Creator comments removed: {stats['creator_comments_removed']:,}")
    print(f"Reply comments in input: {stats['reply_comments_input']:,}")
    print(f"Reply comments removed by top-level filter: {stats['reply_comments_removed']:,}")
    print(f"Comments after cleaning: {stats['comments_after_cleaning']:,}")
    print(f"Output rows: {stats['output_rows']:,}")


if __name__ == "__main__":
    main()
