#!/usr/bin/env python3
"""
Build rolling four-month Instagram RFE metrics on a monthly sliding window.

Inputs:
- comments: filtered non-creator comments with singleton users removed
- posts: posts in the same date range with at least one available comment in ig_comments_clean

Metrics per user within each rolling window:
- recency: number of posts published after the user's last comment
- frequency: number of distinct posts commented by the user in the window
- engagement: average words per comment in the window, counting contiguous repeats
  of the same emoji as one word and different adjacent emoji as separate words
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

COMMENTS_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_input.csv"
POSTS_PATH = SCRIPT_DIR / "ig_posts_dynamic_rfe_available_comments.csv"
OUTPUT_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_rolling_metrics.csv"
WINDOWS_OUTPUT_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_rolling_windows.csv"

START_MONTH = pd.Timestamp("2023-01-01")
LAST_WINDOW_START = pd.Timestamp("2025-12-01")
WINDOW_MONTHS = 4

COMMENT_COLUMNS = [
    "comment_id",
    "media_id",
    "text",
    "timestamp",
    "from_id",
    "from_username",
    "user_key",
]

POST_COLUMNS = [
    "media_id",
    "timestamp",
]

TARGET_COLUMNS = [
    "window_id",
    "window_label",
    "window_index",
    "window_start",
    "window_end",
    "window_total_days",
    "posts_in_window",
    "user_key",
    "user_id",
    "username",
    "comment_count",
    "last_comment_timestamp",
    "recency",
    "frequency",
    "engagement",
]

NON_EMOJI_TOKEN_RE = re.compile(
    r"https?://\S+|[#@]?\w+(?:[.'â€™_-]\w+)*",
    flags=re.UNICODE,
)

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


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    window_label: str
    window_index: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def end_exclusive(self) -> pd.Timestamp:
        return self.end + pd.Timedelta(days=1)

    @property
    def total_days(self) -> int:
        return int((self.end - self.start).days + 1)


def is_emojiish_char(char: str) -> bool:
    codepoint = ord(char)
    if codepoint in EMOJI_CONNECTORS or 0x1F3FB <= codepoint <= 0x1F3FF:
        return True
    for start, end in EMOJI_RANGES:
        if start <= codepoint <= end:
            return True
    return False


def is_regional_indicator(codepoint: int) -> bool:
    return 0x1F1E6 <= codepoint <= 0x1F1FF


def is_skin_tone_modifier(codepoint: int) -> bool:
    return 0x1F3FB <= codepoint <= 0x1F3FF


def is_emoji_base_char(char: str) -> bool:
    codepoint = ord(char)
    if codepoint in EMOJI_CONNECTORS or is_skin_tone_modifier(codepoint):
        return False
    if unicodedata.combining(char) > 0:
        return False
    return is_emojiish_char(char)


def is_combining_or_extender(char: str) -> bool:
    codepoint = ord(char)
    return (
        unicodedata.combining(char) > 0
        or codepoint in {0x20E3, 0xFE0E, 0xFE0F}
        or is_skin_tone_modifier(codepoint)
    )


def consume_keycap_sequence(value: str, start: int) -> int | None:
    if value[start] not in KEYCAP_BASE_CHARS:
        return None

    end = start + 1
    if end < len(value) and ord(value[end]) == 0xFE0F:
        end += 1

    if end < len(value) and ord(value[end]) == 0x20E3:
        return end + 1

    return None


def consume_emoji_cluster(value: str, start: int) -> tuple[str, int] | None:
    keycap_end = consume_keycap_sequence(value, start)
    if keycap_end is not None:
        return value[start:keycap_end], keycap_end

    if start >= len(value):
        return None

    char = value[start]
    codepoint = ord(char)
    if is_regional_indicator(codepoint):
        end = start + 1
        if end < len(value) and is_regional_indicator(ord(value[end])):
            end += 1
        return value[start:end], end

    if not is_emoji_base_char(char):
        return None

    end = start + 1
    while end < len(value) and is_combining_or_extender(value[end]):
        end += 1

    while end < len(value) and ord(value[end]) == 0x200D:
        end += 1
        if end >= len(value):
            break
        end += 1
        while end < len(value) and is_combining_or_extender(value[end]):
            end += 1

    return value[start:end], end


def count_words_with_emoji(text: object) -> int:
    if pd.isna(text):
        return 0

    value = str(text).strip()
    if not value:
        return 0

    count = 0
    last_emoji_cluster: str | None = None
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
            last_emoji_cluster = None
            index += 1
            continue

        emoji_cluster = consume_emoji_cluster(value, index)
        if emoji_cluster is not None:
            count += flush_non_emoji()
            cluster_text, next_index = emoji_cluster
            if cluster_text != last_emoji_cluster:
                count += 1
            last_emoji_cluster = cluster_text
            index = next_index
            continue

        last_emoji_cluster = None
        non_emoji_chunk.append(char)
        index += 1

    count += flush_non_emoji()
    return count


def build_window_specs() -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    current_start = START_MONTH
    index = 1

    while current_start <= LAST_WINDOW_START:
        current_end = current_start + pd.DateOffset(months=WINDOW_MONTHS) - pd.Timedelta(days=1)
        label = f"{current_start.strftime('%Y-%m')} to {current_end.strftime('%Y-%m')}"
        windows.append(
            WindowSpec(
                window_id=f"w{index:02d}",
                window_label=label,
                window_index=index,
                start=current_start,
                end=pd.Timestamp(current_end),
            )
        )
        current_start = current_start + pd.offsets.MonthBegin(1)
        index += 1

    return windows


def load_comments() -> pd.DataFrame:
    comments = pd.read_csv(
        COMMENTS_PATH,
        usecols=COMMENT_COLUMNS,
        dtype={
            "comment_id": "string",
            "media_id": "string",
            "text": "string",
            "timestamp": "string",
            "from_id": "string",
            "from_username": "string",
            "user_key": "string",
        },
    )
    comments["timestamp"] = pd.to_datetime(comments["timestamp"], errors="coerce")
    comments = comments.dropna(subset=["comment_id", "media_id", "timestamp", "user_key"]).copy()
    comments["text"] = comments["text"].fillna("").astype("string")
    comments["from_username"] = comments["from_username"].fillna("").astype("string").str.strip()
    comments["from_id"] = comments["from_id"].fillna("").astype("string").str.strip()
    comments.loc[comments["from_id"] == "", "from_id"] = pd.NA
    comments["word_count"] = comments["text"].map(count_words_with_emoji).astype(int)
    return comments


def load_posts() -> pd.DataFrame:
    posts = pd.read_csv(
        POSTS_PATH,
        usecols=POST_COLUMNS,
        dtype={
            "media_id": "string",
            "timestamp": "string",
        },
    )
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")
    posts = posts.dropna(subset=["media_id", "timestamp"]).copy()
    posts = posts.sort_values(["timestamp", "media_id"], kind="mergesort").reset_index(drop=True)
    return posts


def build_window_metrics(window: WindowSpec, comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    posts_mask = (posts["timestamp"] >= window.start) & (posts["timestamp"] < window.end_exclusive)
    comments_mask = (comments["timestamp"] >= window.start) & (comments["timestamp"] < window.end_exclusive)

    window_posts = posts.loc[posts_mask].copy()
    window_comments = comments.loc[comments_mask].copy()
    posts_in_window = int(len(window_posts))

    if posts_in_window == 0 or window_comments.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    window_comments = window_comments.sort_values(
        ["user_key", "timestamp", "comment_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    posts_seen = np.searchsorted(
        window_posts["timestamp"].to_numpy(dtype="datetime64[ns]"),
        window_comments["timestamp"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    window_comments["posts_seen_by_comment_time"] = posts_seen

    aggregated = (
        window_comments.groupby("user_key", sort=False)
        .agg(
            user_id=("from_id", "last"),
            username=("from_username", "last"),
            comment_count=("comment_id", "size"),
            frequency=("media_id", "nunique"),
            last_comment_timestamp=("timestamp", "max"),
            last_posts_seen=("posts_seen_by_comment_time", "last"),
            engagement=("word_count", "mean"),
        )
        .reset_index()
    )

    aggregated["frequency"] = pd.to_numeric(aggregated["frequency"], errors="coerce").astype(int)
    aggregated["recency"] = (posts_in_window - aggregated["last_posts_seen"]).astype(int)
    aggregated["window_id"] = window.window_id
    aggregated["window_label"] = window.window_label
    aggregated["window_index"] = window.window_index
    aggregated["window_start"] = window.start.date().isoformat()
    aggregated["window_end"] = window.end.date().isoformat()
    aggregated["window_total_days"] = window.total_days
    aggregated["posts_in_window"] = posts_in_window

    return aggregated[TARGET_COLUMNS]


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    comments = load_comments()
    posts = load_posts()
    windows = build_window_specs()

    metrics_frames = [build_window_metrics(window, comments, posts) for window in windows]
    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics = metrics.sort_values(["window_index", "user_key"], kind="mergesort").reset_index(drop=True)

    windows_df = pd.DataFrame(
        {
            "window_id": [window.window_id for window in windows],
            "window_label": [window.window_label for window in windows],
            "window_index": [window.window_index for window in windows],
            "window_start": [window.start.date().isoformat() for window in windows],
            "window_end": [window.end.date().isoformat() for window in windows],
            "window_total_days": [window.total_days for window in windows],
            "posts_in_window": [
                int(
                    (
                        (posts["timestamp"] >= window.start)
                        & (posts["timestamp"] < window.end_exclusive)
                    ).sum()
                )
                for window in windows
            ],
        }
    )
    return metrics, windows_df


def main() -> None:
    metrics, windows_df = build_outputs()
    metrics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    windows_df.to_csv(WINDOWS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(metrics):,} rows to {OUTPUT_PATH}")
    print(f"Saved {len(windows_df):,} rows to {WINDOWS_OUTPUT_PATH}")
    print(f"Rolling windows: {len(windows_df)}")
    print(f"First window: {windows_df.iloc[0]['window_start']} to {windows_df.iloc[0]['window_end']}")
    print(f"Last window: {windows_df.iloc[-1]['window_start']} to {windows_df.iloc[-1]['window_end']}")


if __name__ == "__main__":
    main()
