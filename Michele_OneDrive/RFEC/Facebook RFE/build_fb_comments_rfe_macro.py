#!/usr/bin/env python3
"""
Build annual user RFE metrics for Facebook comments.

Cleaning choices applied before the metrics:
- keep only non-creator comments
- optionally keep only top-level comments
- use `from_id` when available and fall back to a name-based key otherwise

Metrics per user within each era:
- recency: posts published after the user's last comment, divided by total era posts
- frequency: selected user activity count divided by Facebook posts published in the era
- engagement: average words per comment, counting any contiguous emoji run as 1 word
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

CREATOR_NAMES = {"camihawke", "camilla boniardi"}
CREATOR_IDS = {"1686085491707042", "1396485587053253"}
TOP_LEVEL_PARENT_ID = "zero_lvl_comment"

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
]
FINAL_MIN_COMMENTS_PER_ERA = 2

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


def build_annual_era_specs() -> list[EraSpec]:
    era_specs = [
        EraSpec(str(year), pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        for year in range(2016, 2026)
    ]
    era_specs.append(
        EraSpec("2026", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
    )
    return era_specs


FINAL_ERA_LABELS = ("2019", "2020", "2021", "2022", "2026")
FINAL_ERA_SPECS = [era for era in build_annual_era_specs() if era.label in FINAL_ERA_LABELS]
FINAL_ERA_ORDER = [era.label for era in FINAL_ERA_SPECS]
GLOBAL_START = FINAL_ERA_SPECS[0].start
GLOBAL_END_EXCLUSIVE = FINAL_ERA_SPECS[-1].end_exclusive

EMOJI_RANGES = (
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
    0x200D,
    0x20E3,
    0xFE0E,
    0xFE0F,
}

KEYCAP_BASE_CHARS = set("0123456789#*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the annual Facebook comments RFE macro from cleaned comments."
    )
    parser.add_argument(
        "--comments",
        default=WORKSPACE_DIR / "fb_comments_clean.csv",
        type=Path,
        help="Path to the Facebook comments CSV.",
    )
    parser.add_argument(
        "--posts",
        default=WORKSPACE_DIR / "fb_posts_clean.csv",
        type=Path,
        help="Path to the Facebook posts CSV.",
    )
    parser.add_argument(
        "--output",
        default=SCRIPT_DIR / "fb_comments_RFE_macro_top_level_annual_2017_2022_2026_min2_absolute_rf.csv",
        type=Path,
        help="Output macro CSV path.",
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


def assign_era(
    timestamps: pd.Series,
    era_specs: list[EraSpec],
    era_order: list[str],
) -> pd.Categorical:
    eras = pd.Series(pd.NA, index=timestamps.index, dtype="object")
    for era in era_specs:
        mask = (timestamps >= era.start) & (timestamps < era.end_exclusive)
        eras.loc[mask] = era.label
    return pd.Categorical(eras, categories=era_order, ordered=True)


def load_posts(
    path: Path,
    era_specs: list[EraSpec],
    era_order: list[str],
    global_start: pd.Timestamp,
    global_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    posts = pd.read_csv(
        path,
        usecols=["post_id", "created_date"],
        dtype={"post_id": "string", "created_date": "string"},
    )
    posts["timestamp"] = pd.to_datetime(posts["created_date"], errors="coerce")
    posts = posts.dropna(subset=["post_id", "timestamp"]).copy()
    posts = posts[(posts["timestamp"] >= global_start) & (posts["timestamp"] < global_end_exclusive)]
    posts["era"] = assign_era(posts["timestamp"], era_specs, era_order)
    posts = posts.dropna(subset=["era"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return posts[["post_id", "timestamp", "era"]]


def load_comments(
    path: Path,
    era_specs: list[EraSpec],
    era_order: list[str],
    global_start: pd.Timestamp,
    global_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    comments = pd.read_csv(
        path,
        usecols=[
            "comment_id",
            "post_id",
            "message",
            "created_time",
            "from_name",
            "from_id",
            "parent_id",
        ],
        dtype={
            "comment_id": "string",
            "post_id": "string",
            "message": "string",
            "created_time": "string",
            "from_name": "string",
            "from_id": "string",
            "parent_id": "string",
        },
    )

    comments["timestamp"] = pd.to_datetime(comments["created_time"], errors="coerce", utc=True).dt.tz_localize(None)
    comments = comments.dropna(subset=["comment_id", "post_id", "timestamp", "from_name"]).copy()
    comments = comments[
        (comments["timestamp"] >= global_start) & (comments["timestamp"] < global_end_exclusive)
    ]

    comments["message"] = comments["message"].fillna("").astype("string")
    comments["from_name"] = comments["from_name"].fillna("").astype("string").str.strip()
    comments["from_name_norm"] = comments["from_name"].str.casefold()
    comments["from_id"] = comments["from_id"].fillna("").astype("string").str.strip()
    comments.loc[comments["from_id"] == "", "from_id"] = pd.NA
    comments["parent_id"] = comments["parent_id"].fillna("").astype("string").str.strip()

    creator_mask = comments["from_name_norm"].isin(CREATOR_NAMES) | comments["from_id"].isin(CREATOR_IDS)
    comments = comments.loc[~creator_mask].copy()

    comments["user_key"] = comments["from_id"].fillna("name:" + comments["from_name_norm"])
    comments["era"] = assign_era(comments["timestamp"], era_specs, era_order)
    comments = comments.dropna(subset=["era"]).copy()
    comments["word_count"] = comments["message"].map(count_words_with_emoji).astype(int)
    return comments


def keep_top_level_comments(comments: pd.DataFrame) -> pd.DataFrame:
    top_level_mask = comments["parent_id"].fillna("").eq(TOP_LEVEL_PARENT_ID)
    return comments.loc[top_level_mask].copy()


def build_era_metrics(
    era: EraSpec,
    comments: pd.DataFrame,
    posts: pd.DataFrame,
) -> pd.DataFrame:
    era_comments = comments.loc[comments["era"] == era.label].copy()
    era_posts = posts.loc[posts["era"] == era.label].copy()
    posts_in_era = int(len(era_posts))

    if posts_in_era == 0:
        raise ValueError(f"No posts found for era {era.label}.")

    user_counts = era_comments.groupby("user_key").size().rename("comment_count")
    eligible_users = user_counts[user_counts >= FINAL_MIN_COMMENTS_PER_ERA]
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

    aggregated = (
        era_comments.groupby("user_key", sort=False)
        .agg(
            user_id=("from_id", "last"),
            username=("from_name", "last"),
            comment_count=("comment_id", "size"),
            distinct_posts_count=("post_id", "nunique"),
            last_comment_timestamp=("timestamp", "max"),
            last_posts_seen=("posts_seen_by_comment_time", "last"),
            engagement=("word_count", "mean"),
        )
        .reset_index()
    )

    aggregated["recency_posts_since_last_comment"] = (
        posts_in_era - aggregated["last_posts_seen"]
    ).astype(int)
    aggregated["recency"] = aggregated["recency_posts_since_last_comment"] / posts_in_era
    aggregated["frequency"] = pd.to_numeric(aggregated["distinct_posts_count"], errors="coerce")
    aggregated["era"] = era.label
    aggregated["era_start"] = era.start.date().isoformat()
    aggregated["era_end"] = era.end.date().isoformat()
    aggregated["era_total_days"] = era.total_days
    aggregated["posts_in_era"] = posts_in_era

    return aggregated[TARGET_COLUMNS]


def build_macro(
    comments_path: Path,
    posts_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    comments_raw = pd.read_csv(
        comments_path,
        usecols=["from_name", "from_id", "parent_id"],
        dtype={
            "from_name": "string",
            "from_id": "string",
            "parent_id": "string",
        },
    )
    name_norm = comments_raw["from_name"].fillna("").astype("string").str.strip().str.casefold()
    id_norm = comments_raw["from_id"].fillna("").astype("string").str.strip()
    creator_mask = name_norm.isin(CREATOR_NAMES) | id_norm.isin(CREATOR_IDS)
    reply_mask = comments_raw["parent_id"].fillna("").astype("string").str.strip().ne(TOP_LEVEL_PARENT_ID)
    stats = {
        "input_comments": int(len(comments_raw)),
        "creator_comments_removed": int(creator_mask.sum()),
        "reply_comments_input": int(reply_mask.sum()),
    }
    del comments_raw

    posts = load_posts(posts_path, FINAL_ERA_SPECS, FINAL_ERA_ORDER, GLOBAL_START, GLOBAL_END_EXCLUSIVE)
    comments = load_comments(comments_path, FINAL_ERA_SPECS, FINAL_ERA_ORDER, GLOBAL_START, GLOBAL_END_EXCLUSIVE)
    stats["reply_comments_removed"] = int(comments["parent_id"].fillna("").ne(TOP_LEVEL_PARENT_ID).sum())
    comments = keep_top_level_comments(comments)
    stats["comments_after_cleaning"] = int(len(comments))

    results = [
        build_era_metrics(era, comments, posts)
        for era in FINAL_ERA_SPECS
    ]

    macro = pd.concat(results, ignore_index=True)
    macro["era"] = pd.Categorical(macro["era"], categories=FINAL_ERA_ORDER, ordered=True)
    macro = macro.sort_values(["era", "user_key"], kind="mergesort").reset_index(drop=True)
    macro["era"] = macro["era"].astype("string")
    stats["output_rows"] = int(len(macro))
    return macro, stats


def main() -> None:
    args = parse_args()
    macro, stats = build_macro(args.comments, args.posts)
    macro.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Eras kept: {', '.join(FINAL_ERA_ORDER)}")
    print(f"Top-level only: True")
    print(f"Minimum comments per era: {FINAL_MIN_COMMENTS_PER_ERA}")
    print("Recency: scaled by posts in year")
    print("Frequency: absolute distinct posts in year")
    print(f"Saved {len(macro):,} rows to {args.output}")
    print(f"Input comments: {stats['input_comments']:,}")
    print(f"Creator comments removed: {stats['creator_comments_removed']:,}")
    print(f"Reply comments in input: {stats['reply_comments_input']:,}")
    print(f"Reply comments removed by top-level filter: {stats['reply_comments_removed']:,}")
    print(f"Comments after cleaning: {stats['comments_after_cleaning']:,}")
    print(f"Output rows: {stats['output_rows']:,}")


if __name__ == "__main__":
    main()
