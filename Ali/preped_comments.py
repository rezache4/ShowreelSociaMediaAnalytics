"""Split the prepped comment exports per platform and fold their features into
an EXTENDED community-vibe schema.

The Data Preparation Pipeline emits three combined files in ``Preped_Comments/``:

    comments_llm.jsonl   — raw text per comment       (LLM / CAG input)
    comments_ml.parquet  — numeric feature matrix      (emoji/punctuation stats)
    comments_gml.parquet — directed edge list          (reply structure)

All three share ``comment_id`` and carry a ``platform`` field. This module:

  1. ``separate_by_platform`` — fan the three files out into
     ``Preped_Comments/by_platform/<platform>/comments_{llm.jsonl,ml.parquet,gml.parquet}``
     so each platform can be processed independently (today: instagram only;
     FB / TikTok / YouTube slot in automatically once present).

  2. ``PrepedCommentsLoader.load_for_pipeline`` — rebuild a raw-schema comment
     DataFrame (text + ids + reply ref) that drops straight into
     ``InstagramMultimodalPipeline(external_comments=...)`` /
     ``community_vibe_pipeline`` — no recompute needed.

  3. ``aggregate_community_features`` — turn the *precomputed* ML/GML features
     into per-``media_id`` deterministic signals that EXTEND the LLM community
     vibe (emoji intensity/diversity, reply depth, interrogative/exclamatory
     mix, link/mention/hashtag pressure, author concentration). These are cheap,
     exact, and complement the probabilistic ``sentiment_polarization_index`` /
     ``topical_adherence_score`` from the Gemini phases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("preped_comments")

# Raw IG-compatible column names (so the base SchemaNormalizer accepts them).
COL_COMMENT_ID = "comment_id"
COL_MEDIA_ID = "media_id"
COL_TEXT = "text"
COL_AUTHOR = "author_id"
COL_REPLY = "reply_to_comment_id"
COL_TIMESTAMP = "timestamp"
COL_PLATFORM = "platform"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrepedConfig:
    input_dir: str = "Preped_Comments"
    output_dir: str = "Preped_Comments/by_platform"
    llm_name: str = "comments_llm.jsonl"
    ml_name: str = "comments_ml.parquet"
    gml_name: str = "comments_gml.parquet"


# --------------------------------------------------------------------------- #
# Extended community-vibe schema (deterministic, from prepped features)
# --------------------------------------------------------------------------- #
@dataclass
class ExtendedCommunityVibeSchema:
    """The per-media columns produced by ``aggregate_community_features``.

    These EXTEND the LLM community-vibe fields (sentiment_polarization_index,
    topical_adherence_score, dominant_community_emotion) with exact, cheap
    signals derived from the prepped ML/GML comment features.
    """

    columns: List[str] = field(
        default_factory=lambda: [
            "comment_volume",          # number of comments on the media
            "unique_authors",          # distinct commenters
            "author_concentration",    # 1 - unique_authors/comment_volume (0=all distinct)
            "reply_ratio",             # share of comments that are replies (GML edges)
            "mean_word_count",         # avg comment length (words)
            "mean_emoji_per_word",     # emoji intensity
            "mean_emoji_entropy",      # emoji diversity (Shannon)
            "mean_emoji_variety",      # unique/total emoji ratio
            "interrogative_ratio",     # share of comments with a '?'
            "exclamatory_ratio",       # share of comments with a '!'
            "link_ratio",              # share of comments containing a URL
            "mention_ratio",           # share with @mention
            "hashtag_ratio",           # share with #hashtag
        ]
    )


# --------------------------------------------------------------------------- #
# 1) Separation per platform
# --------------------------------------------------------------------------- #
def separate_by_platform(config: Optional[PrepedConfig] = None) -> Dict[str, Dict[str, str]]:
    """Fan the combined exports out into one folder per platform.

    Returns a manifest ``{platform: {kind: path, "rows": n}}`` and writes a
    ``manifest.json`` alongside the splits.
    """
    cfg = config or PrepedConfig()
    in_dir, out_dir = Path(cfg.input_dir), Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, str]] = {}

    # --- LLM jsonl: stream so we never hold 100MB+ in memory ---------------
    llm_path = in_dir / cfg.llm_name
    if llm_path.exists():
        handles: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        try:
            with llm_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        plat = json.loads(line).get(COL_PLATFORM, "unknown")
                    except json.JSONDecodeError:
                        plat = "unknown"
                    if plat not in handles:
                        pdir = out_dir / plat
                        pdir.mkdir(parents=True, exist_ok=True)
                        handles[plat] = (pdir / cfg.llm_name).open("w", encoding="utf-8")
                        counts[plat] = 0
                    handles[plat].write(line + "\n")
                    counts[plat] += 1
        finally:
            for h in handles.values():
                h.close()
        for plat, n in counts.items():
            manifest.setdefault(plat, {})["llm"] = str(out_dir / plat / cfg.llm_name)
            manifest[plat]["llm_rows"] = str(n)
        LOGGER.info("LLM split: %s", {p: c for p, c in counts.items()})
    else:
        LOGGER.warning("Missing %s", llm_path)

    # --- ML & GML parquet: groupby platform, write per group ---------------
    for kind, name in (("ml", cfg.ml_name), ("gml", cfg.gml_name)):
        ppath = in_dir / name
        if not ppath.exists():
            LOGGER.warning("Missing %s", ppath)
            continue
        df = pd.read_parquet(ppath)
        if COL_PLATFORM not in df.columns:
            LOGGER.warning("%s has no '%s' column; writing as 'unknown'.", name, COL_PLATFORM)
            df[COL_PLATFORM] = "unknown"
        for plat, sub in df.groupby(COL_PLATFORM):
            pdir = out_dir / str(plat)
            pdir.mkdir(parents=True, exist_ok=True)
            dest = pdir / name
            sub.drop(columns=[COL_PLATFORM]).to_parquet(dest, compression="snappy", index=False)
            manifest.setdefault(str(plat), {})[kind] = str(dest)
            manifest[str(plat)][f"{kind}_rows"] = str(len(sub))
        LOGGER.info("%s split: %s", kind, df[COL_PLATFORM].value_counts().to_dict())

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest → %s", out_dir / "manifest.json")
    return manifest


# --------------------------------------------------------------------------- #
# 2 + 3) Per-platform loading & extended-vibe aggregation
# --------------------------------------------------------------------------- #
class PrepedCommentsLoader:
    """Read a platform's split and expose pipeline-ready / aggregate views."""

    def __init__(self, config: Optional[PrepedConfig] = None) -> None:
        self.cfg = config or PrepedConfig()

    def _pdir(self, platform: str) -> Path:
        return Path(self.cfg.output_dir) / platform

    def _read_llm(self, platform: str) -> pd.DataFrame:
        path = self._pdir(platform) / self.cfg.llm_name
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append(rec)
        df = pd.DataFrame(rows)
        return df

    def load_for_pipeline(
        self, platform: str, media_ids: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Raw-schema comments (text + ids + reply ref) for the vibe pipelines.

        Merges LLM text with the GML edge list so each comment carries its
        ``media_id`` and ``reply_to_comment_id``. Output columns are named so the
        base ``SchemaNormalizer.normalize_comments(df, platform)`` accepts them.
        """
        gml = pd.read_parquet(self._pdir(platform) / self.cfg.gml_name)
        keep = set(map(str, media_ids)) if media_ids is not None else None
        if keep is not None:
            gml = gml[gml[COL_MEDIA_ID].astype(str).isin(keep)]

        llm = self._read_llm(platform)
        merged = gml.merge(
            llm[[COL_COMMENT_ID, COL_TEXT]], on=COL_COMMENT_ID, how="left"
        )
        # author_id present in both; prefer GML's. reply_to_comment_id → parent.
        out = merged[[COL_COMMENT_ID, COL_MEDIA_ID, COL_TEXT, COL_AUTHOR,
                      COL_REPLY, COL_TIMESTAMP]].copy()
        out[COL_PLATFORM] = platform
        LOGGER.info("[%s] pipeline comments: %d rows across %d media.",
                    platform, len(out), out[COL_MEDIA_ID].nunique())
        return out

    def aggregate_community_features(
        self, platform: str, media_ids: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Per-``media_id`` extended community-vibe features (deterministic)."""
        ml = pd.read_parquet(self._pdir(platform) / self.cfg.ml_name)
        gml = pd.read_parquet(self._pdir(platform) / self.cfg.gml_name)
        if media_ids is not None:
            keep = set(map(str, media_ids))
            ml = ml[ml[COL_MEDIA_ID].astype(str).isin(keep)]
            gml = gml[gml[COL_MEDIA_ID].astype(str).isin(keep)]

        # Boolean helpers from the ML feature columns.
        ml = ml.assign(
            _has_q=(ml.get("question_count", 0) > 0).astype(int),
            _has_excl=(ml.get("exclamation_count", 0) > 0).astype(int),
            _has_mention=(ml.get("mention_count", 0) > 0).astype(int),
            _has_hashtag=(ml.get("hashtag_count", 0) > 0).astype(int),
        )
        g = ml.groupby(COL_MEDIA_ID)
        agg = pd.DataFrame({
            "comment_volume": g.size(),
            "unique_authors": g[COL_AUTHOR].nunique(),
            "mean_word_count": g["word_count"].mean(),
            "mean_emoji_per_word": g["emoji_per_word_ratio"].mean(),
            "mean_emoji_entropy": g["emoji_entropy"].mean(),
            "mean_emoji_variety": g["emoji_variety_ratio"].mean(),
            "interrogative_ratio": g["_has_q"].mean(),
            "exclamatory_ratio": g["_has_excl"].mean(),
            "link_ratio": g["has_links"].mean(),
            "mention_ratio": g["_has_mention"].mean(),
            "hashtag_ratio": g["_has_hashtag"].mean(),
        })
        agg["author_concentration"] = 1.0 - (agg["unique_authors"] / agg["comment_volume"]).clip(0, 1)

        # Reply structure comes from the GML edges (reply_to_comment_id not null).
        reply_share = (
            gml.assign(_is_reply=gml[COL_REPLY].notna().astype(int))
            .groupby(COL_MEDIA_ID)["_is_reply"].mean()
            .rename("reply_ratio")
        )
        agg = agg.join(reply_share, how="left")
        agg["reply_ratio"] = agg["reply_ratio"].fillna(0.0)

        # Order to the documented schema + downcast.
        cols = ExtendedCommunityVibeSchema().columns
        agg = agg.reset_index()[[COL_MEDIA_ID, *cols]]
        float_cols = agg.select_dtypes("float").columns
        agg[float_cols] = agg[float_cols].apply(pd.to_numeric, downcast="float")
        LOGGER.info("[%s] extended community features for %d media.", platform, len(agg))
        return agg


# --------------------------------------------------------------------------- #
# Main — separate, then preview the extended vibe for a few media.
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    manifest = separate_by_platform()
    LOGGER.info("Manifest:\n%s", json.dumps(manifest, indent=2))

    loader = PrepedCommentsLoader()
    for platform in manifest:
        ext = loader.aggregate_community_features(platform)
        top = ext.sort_values("comment_volume", ascending=False).head(5)
        pd.set_option("display.max_columns", None, "display.width", 240)
        LOGGER.info("[%s] extended community-vibe (top-5 by volume):\n%s",
                    platform, top.to_string(index=False))


if __name__ == "__main__":
    main()
