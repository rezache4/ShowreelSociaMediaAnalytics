"""Show Reel — Instagram MULTIMODAL Post Enrichment + Cache-Augmented Comment Vibe.

Instagram-specific superset of ``community_vibe_pipeline.py``. It does everything
the base pipeline does (Phase 1 context, Phase 2 local NLP, Phase 3 community
vibe) but grounds the analysis in the **actual media content** you extracted:

    * Reels  →  multimodal_dataset/<shortcode>/
                 ├─ <shortcode>.info.json   (yt-dlp metadata + caption + comments)
                 ├─ transcription.txt        (Whisper transcript)
                 └─ frames/*.jpg             (extracted scene/first/second frames)

    * Images / Carousels  →  multimodal_dataset_fixed/<type>/<shortcode>/
                 ├─ <ts>_UTC.json            (instaloader node metadata)
                 └─ <ts>_UTC[_n].jpg         (the still frame(s))

Pipeline
--------
    Phase 1 (multimodal)  — Gemini reads FRAMES + transcript + caption to extract
                            format_type / primary_topic / tone / brand_entities
                            plus visual_summary / on_screen_text / visual_setting.

    Phase 2 (local NLP)   — reused verbatim from the base pipeline (spaCy):
                            token metrics, entity overlap, cosine similarity.

    Phase 3 (CAG)         — Cache-Augmented Generation: the post's heavy
                            multimodal context (frames + transcript + caption +
                            Phase-1 summary) is cached ONCE in Vertex
                            ``CachedContent``; the comments are then scored in
                            reusable chunked queries against that single cache —
                            so the frames are uploaded/processed once, not per
                            comment batch. Falls back to inline context if the
                            context is below the cache minimum or caching fails.

Output: ``enriched_ig_multimodal_vibe_matrix.parquet``.

Heavy deps (vertexai, spaCy) are import-guarded; with ``enable_llm=False`` the
whole thing runs locally on the real folders using deterministic fallbacks.

Author: AFB_Lab — Lead Data Science / MLOps
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prompts import (
    CAG_COMMENT_CHUNK_PROMPT,
    CAG_SYSTEM_INSTRUCTION,
    MULTIMODAL_POST_PROMPT,
)

# --------------------------------------------------------------------------- #
# Reuse the verified base pipeline (same directory module).
# --------------------------------------------------------------------------- #
from community_vibe_pipeline import (  # noqa: E402
    ALLOWED_FORMATS,
    ALLOWED_TONES,
    CANON_AUTHOR_ID,
    CANON_COMMENT_ID,
    CANON_COMMENT_TEXT,
    CANON_MEDIA_ID,
    CANON_PARENT_ID,
    CANON_PLATFORM,
    CANON_POST_TEXT,
    CANON_TIMESTAMP,
    CANON_TRANSCRIPT,
    COMMUNITY_VIBE_RESPONSE_SCHEMA,
    LocalNLPEnricher,
    PipelineConfig,
    PostContext,
    SchemaNormalizer,
    _HAS_PYDANTIC,
)

# --------------------------------------------------------------------------- #
# Optional Vertex AI multimodal + context-caching layer.
# --------------------------------------------------------------------------- #
try:
    import vertexai
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel,
        Part,
    )
    from vertexai.preview import caching as _caching

    _HAS_VERTEX_MM = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_VERTEX_MM = False

if _HAS_PYDANTIC:
    from pydantic import BaseModel, Field, ValidationError, field_validator


LOGGER = logging.getLogger("instagram_multimodal_pipeline")

# Frame filename heuristic ranking (scene/first/second extractions).
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IGMultimodalConfig(PipelineConfig):
    """Base config + Instagram-multimodal + CAG knobs."""

    # --- on-disk roots ------------------------------------------------------
    mm_reel_root: str = "multimodal_dataset"          # reels w/ frames+transcript
    mm_fixed_root: str = "multimodal_dataset_fixed"   # instaloader image/carousel

    # --- frame sampling -----------------------------------------------------
    max_frames: int = 6                # frames fed to the model per post
    frame_max_edge_px: int = 0         # 0 = send as-is (no resize dependency)

    # --- Cache-Augmented Generation ----------------------------------------
    enable_cache: bool = True          # build a CachedContent per post
    cache_ttl_minutes: int = 15        # keep the cache warm for chunked queries
    comment_chunk_size: int = 40       # comments per CAG query (reuses cache)

    # --- discovery / sampling ----------------------------------------------
    max_posts: Optional[int] = None    # cap posts (smoke runs); None = all
    include_embedded_comments: bool = True

    # --- cleaned-corpus integration (Data/ig_*_cleaned.parquet) -------------
    # permalink in ig_posts_cleaned carries the shortcode → numeric media_id,
    # the bridge to the full ig_comments_cleaned corpus.
    ig_posts_parquet: Optional[str] = None       # e.g. "../Data/ig_posts_cleaned.parquet"
    ig_comments_parquet: Optional[str] = None     # e.g. "../Data/ig_comments_cleaned.parquet"

    # --- output -------------------------------------------------------------
    output_path: str = "enriched_ig_multimodal_vibe_matrix.parquet"


# --------------------------------------------------------------------------- #
# Structured-output schemas (multimodal extensions)
# --------------------------------------------------------------------------- #
if _HAS_PYDANTIC:

    class MultimodalPostContext(PostContext):
        """Phase-1 contract enriched with vision-grounded fields."""

        visual_summary: str = Field("", description="What is literally shown across frames.")
        on_screen_text: List[str] = Field(default_factory=list, description="Visible text/captions.")
        visual_setting: str = Field("", description="Scene/location archetype.")

        @field_validator("on_screen_text", mode="before")
        @classmethod
        def _coerce(cls, v: Any) -> List[str]:
            if v is None:
                return []
            return [v] if isinstance(v, str) and v.strip() else (list(v) if not isinstance(v, str) else [])

    class CAGCommunityVibe(BaseModel):
        """Phase-3 contract produced from the cached multimodal context."""

        sentiment_polarization_index: float = Field(..., ge=0.0, le=1.0)
        dominant_community_emotion: str = Field(...)
        community_noun_phrases: List[str] = Field(default_factory=list)
        visual_reference_ratio: float = Field(
            0.0, ge=0.0, le=1.0,
            description="Fraction of comments referencing what is visible in the frames.",
        )

        @field_validator("sentiment_polarization_index", "visual_reference_ratio", mode="before")
        @classmethod
        def _clamp(cls, v: Any) -> float:
            try:
                return float(min(1.0, max(0.0, float(v))))
            except (TypeError, ValueError):
                return 0.0

        @field_validator("community_noun_phrases", mode="before")
        @classmethod
        def _coerce(cls, v: Any) -> List[str]:
            if v is None:
                return []
            return [v] if isinstance(v, str) and v.strip() else (list(v) if not isinstance(v, str) else [])
else:  # pragma: no cover
    MultimodalPostContext = dict  # type: ignore
    CAGCommunityVibe = dict  # type: ignore


MULTIMODAL_CONTEXT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "format_type": {"type": "STRING", "enum": list(ALLOWED_FORMATS)},
        "primary_topic": {"type": "STRING"},
        "intended_emotional_tone": {"type": "STRING", "enum": list(ALLOWED_TONES)},
        "brand_entities": {"type": "ARRAY", "items": {"type": "STRING"}},
        "visual_summary": {"type": "STRING"},
        "on_screen_text": {"type": "ARRAY", "items": {"type": "STRING"}},
        "visual_setting": {"type": "STRING"},
    },
    "required": [
        "format_type", "primary_topic", "intended_emotional_tone",
        "brand_entities", "visual_summary", "on_screen_text", "visual_setting",
    ],
}

# Phase-3 CAG schema = base community-vibe schema + visual_reference_ratio.
CAG_VIBE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        **COMMUNITY_VIBE_RESPONSE_SCHEMA["properties"],
        "visual_reference_ratio": {"type": "NUMBER"},
    },
    "required": COMMUNITY_VIBE_RESPONSE_SCHEMA["required"] + ["visual_reference_ratio"],
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class IGPost:
    """One discovered Instagram post with its multimodal artifacts."""

    shortcode: str
    media_subtype: str               # reel | image | carousel | feed
    caption: str
    transcript: str
    timestamp: Optional[str]
    like_count: Optional[int]
    comment_count: Optional[int]
    is_sponsored: bool
    media_id_numeric: Optional[str]  # joins to the big ig_comments corpus
    frame_paths: List[Path]
    embedded_comments: List[Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Multimodal loader — walks the extracted-frame folders
# --------------------------------------------------------------------------- #
class IGMultimodalLoader:
    """Discover reels + instaloader posts and parse them into ``IGPost``s."""

    def __init__(self, config: IGMultimodalConfig) -> None:
        self.config = config

    # --- public ------------------------------------------------------------
    def discover(self) -> List[IGPost]:
        posts: Dict[str, IGPost] = {}
        for root_attr in (self.config.mm_reel_root, self.config.mm_fixed_root):
            root = Path(root_attr)
            if not root.exists():
                LOGGER.warning("Root not found, skipping: %s", root.resolve())
                continue
            for jpath in root.rglob("*.json"):
                try:
                    post = self._parse_json(jpath)
                except Exception as exc:  # one bad file must not kill discovery
                    LOGGER.debug("Skipping %s: %s", jpath, exc)
                    post = None
                if post is None:
                    continue
                # Prefer the video/reel record (has transcript) over a still dupe.
                existing = posts.get(post.shortcode)
                if existing is None or (not existing.transcript and post.transcript):
                    posts[post.shortcode] = post

        ordered = sorted(posts.values(), key=lambda p: p.timestamp or "")
        if self.config.max_posts is not None:
            ordered = ordered[: self.config.max_posts]
        LOGGER.info(
            "Discovered %d IG posts (reels w/ transcript: %d).",
            len(ordered), sum(1 for p in ordered if p.transcript),
        )
        return ordered

    def to_frames(self, posts: Sequence[IGPost]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Canonical media + (embedded) comment DataFrames keyed by shortcode."""
        media_rows: List[Dict[str, Any]] = []
        comment_rows: List[Dict[str, Any]] = []
        for p in posts:
            media_rows.append(
                {
                    CANON_MEDIA_ID: p.shortcode,
                    "media_id_numeric": p.media_id_numeric,
                    "media_subtype": p.media_subtype,
                    CANON_POST_TEXT: p.caption,
                    CANON_TRANSCRIPT: p.transcript,
                    CANON_TIMESTAMP: p.timestamp,
                    CANON_PLATFORM: "instagram",
                    "like_count": p.like_count,
                    "comment_count": p.comment_count,
                    "is_sponsored": p.is_sponsored,
                    "n_frames": len(p.frame_paths),
                    "frame_paths": [str(fp) for fp in p.frame_paths],
                }
            )
            if self.config.include_embedded_comments:
                for c in p.embedded_comments:
                    comment_rows.append({**c, CANON_MEDIA_ID: p.shortcode, CANON_PLATFORM: "instagram"})

        media_df = pd.DataFrame(media_rows)
        comments_df = pd.DataFrame(
            comment_rows,
            columns=[CANON_COMMENT_ID, CANON_MEDIA_ID, CANON_PARENT_ID,
                     CANON_AUTHOR_ID, CANON_COMMENT_TEXT, CANON_TIMESTAMP, CANON_PLATFORM],
        )
        return media_df, comments_df

    # --- json parsers ------------------------------------------------------
    def _parse_json(self, jpath: Path) -> Optional[IGPost]:
        with jpath.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "node" in data:        # instaloader
            return self._parse_instaloader(jpath, data["node"])
        if isinstance(data, dict) and ("webpage_url" in data or "extractor" in data):  # yt-dlp
            return self._parse_reel(jpath, data)
        return None

    def _parse_reel(self, jpath: Path, d: Dict[str, Any]) -> IGPost:
        folder = jpath.parent
        shortcode = str(d.get("id") or d.get("display_id") or folder.name)
        transcript = self._read_transcript(folder / "transcription.txt")
        frames = self._collect_frames(folder / "frames")
        comments = [
            {
                CANON_COMMENT_ID: str(c.get("id")),
                CANON_PARENT_ID: c.get("parent") or None,
                CANON_AUTHOR_ID: str(c.get("author_id") or c.get("author") or ""),
                CANON_COMMENT_TEXT: c.get("text"),
                CANON_TIMESTAMP: self._epoch_to_iso(c.get("timestamp")),
            }
            for c in (d.get("comments") or [])
            if c.get("text")
        ]
        return IGPost(
            shortcode=shortcode,
            media_subtype="reel",
            caption=str(d.get("description") or d.get("title") or ""),
            transcript=transcript,
            timestamp=self._epoch_to_iso(d.get("timestamp")),
            like_count=self._as_int(d.get("like_count")),
            comment_count=self._as_int(d.get("comment_count")),
            is_sponsored="#ad" in (d.get("description") or "").lower(),
            media_id_numeric=None,
            frame_paths=frames,
            embedded_comments=comments,
        )

    def _parse_instaloader(self, jpath: Path, node: Dict[str, Any]) -> IGPost:
        folder = jpath.parent
        subtype = self._infer_subtype(folder, node)
        caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
        caption = caption_edges[0]["node"]["text"] if caption_edges else ""
        like = node.get("edge_media_preview_like", {}).get("count")
        ccount = (
            node.get("edge_media_to_parent_comment", {}).get("count")
            or node.get("edge_media_preview_comment", {}).get("count")
        )
        comments: List[Dict[str, Any]] = []
        for edge in node.get("edge_media_to_parent_comment", {}).get("edges", []):
            cn = edge.get("node", {})
            comments.append(
                {
                    CANON_COMMENT_ID: str(cn.get("id")),
                    CANON_PARENT_ID: None,
                    CANON_AUTHOR_ID: str(cn.get("owner", {}).get("id", "")),
                    CANON_COMMENT_TEXT: cn.get("text"),
                    CANON_TIMESTAMP: self._epoch_to_iso(cn.get("created_at")),
                }
            )
            for rep in cn.get("edge_threaded_comments", {}).get("edges", []):
                rn = rep.get("node", {})
                comments.append(
                    {
                        CANON_COMMENT_ID: str(rn.get("id")),
                        CANON_PARENT_ID: str(cn.get("id")),
                        CANON_AUTHOR_ID: str(rn.get("owner", {}).get("id", "")),
                        CANON_COMMENT_TEXT: rn.get("text"),
                        CANON_TIMESTAMP: self._epoch_to_iso(rn.get("created_at")),
                    }
                )
        return IGPost(
            shortcode=str(node.get("shortcode") or folder.name),
            media_subtype=subtype,
            caption=caption,
            transcript="",  # stills have no transcript
            timestamp=self._epoch_to_iso(node.get("taken_at_timestamp")),
            like_count=self._as_int(like),
            comment_count=self._as_int(ccount),
            is_sponsored=bool(node.get("is_paid_partnership") or node.get("is_ad")),
            media_id_numeric=str(node.get("id")) if node.get("id") else None,
            frame_paths=self._collect_frames(folder),
            embedded_comments=[c for c in comments if c.get(CANON_COMMENT_TEXT)],
        )

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _infer_subtype(folder: Path, node: Dict[str, Any]) -> str:
        for part in folder.parts[::-1]:
            if part in ("reel", "image", "carousel", "feed"):
                return part
        typename = node.get("__typename", "")
        return {"GraphSidecar": "carousel", "GraphVideo": "reel", "GraphImage": "image"}.get(typename, "image")

    def _collect_frames(self, frames_dir: Path) -> List[Path]:
        if not frames_dir.exists():
            return []
        imgs = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        return self._sample_even(imgs, self.config.max_frames)

    @staticmethod
    def _sample_even(items: List[Path], k: int) -> List[Path]:
        if len(items) <= k:
            return items
        idx = np.linspace(0, len(items) - 1, k).round().astype(int)
        return [items[i] for i in sorted(dict.fromkeys(idx))]

    @staticmethod
    def _read_transcript(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        # Strip "[0.00s -> 2.34s]" timestamps, keep spoken text.
        lines = []
        for line in raw.splitlines():
            txt = line.split("]")[-1].strip() if line.strip().startswith("[") else line.strip()
            if txt:
                lines.append(txt)
        return " ".join(lines)

    @staticmethod
    def _epoch_to_iso(ts: Any) -> Optional[str]:
        try:
            return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _as_int(v: Any) -> Optional[int]:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None


# --------------------------------------------------------------------------- #
# Vertex multimodal client (frames + text) with CAG support
# --------------------------------------------------------------------------- #
class VertexMultimodalClient:
    """Multimodal Gemini wrapper: one-shot generation + Cache-Augmented Generation."""

    def __init__(self, config: IGMultimodalConfig) -> None:
        self.config = config
        self.available = False
        self.call_count = 0
        self.cache_hits = 0
        self.error_count = 0
        self._flash: Optional["GenerativeModel"] = None
        if not (config.enable_llm and _HAS_VERTEX_MM):
            LOGGER.warning(
                "Vertex multimodal disabled/unavailable (enable_llm=%s, sdk=%s); "
                "Phase 1 & 3 use deterministic fallbacks.", config.enable_llm, _HAS_VERTEX_MM,
            )
            return
        try:
            vertexai.init(project=config.gcp_project_id, location=config.gcp_location)
            self._flash = GenerativeModel(config.flash_model)
            self.available = True
            LOGGER.info("Vertex multimodal initialized (%s @ %s).",
                        config.flash_model, config.gcp_location)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Vertex init failed: %s", exc)

    # --- frame loading -----------------------------------------------------
    def _image_parts(self, frame_paths: Sequence[Path]) -> List["Part"]:
        parts: List["Part"] = []
        for fp in frame_paths:
            try:
                mime = "image/png" if fp.suffix.lower() == ".png" else "image/jpeg"
                parts.append(Part.from_data(data=fp.read_bytes(), mime_type=mime))
            except Exception as exc:
                LOGGER.debug("Frame unreadable %s: %s", fp, exc)
        return parts

    # --- Phase 1: one-shot multimodal summary ------------------------------
    def summarize_multimodal(
        self, prompt: str, frame_paths: Sequence[Path]
    ) -> Optional[Dict[str, Any]]:
        if not self.available or self._flash is None:
            return None
        contents = [*self._image_parts(frame_paths), Part.from_text(prompt)]
        cfg = GenerationConfig(
            temperature=self.config.llm_temperature,
            response_mime_type="application/json",
            response_schema=MULTIMODAL_CONTEXT_RESPONSE_SCHEMA,
        )
        for attempt in range(1, self.config.llm_max_retries + 1):
            try:
                self.call_count += 1
                resp = self._flash.generate_content(contents, generation_config=cfg)
                return json.loads(resp.text)
            except Exception as exc:
                LOGGER.warning("Phase-1 multimodal call failed (attempt %d): %s", attempt, exc)
        self.error_count += 1
        return None

    # --- Phase 3: Cache-Augmented Generation -------------------------------
    def build_cache(
        self, shortcode: str, context_text: str, frame_paths: Sequence[Path]
    ) -> Optional[Any]:
        """Cache the post's multimodal context once for reuse across queries."""
        if not (self.available and self.config.enable_cache):
            return None
        try:
            cache = _caching.CachedContent.create(
                model_name=self.config.flash_model,
                system_instruction=CAG_SYSTEM_INSTRUCTION,
                contents=[Part.from_text(context_text), *self._image_parts(frame_paths)],
                ttl=_dt.timedelta(minutes=self.config.cache_ttl_minutes),
                display_name=f"ig-{shortcode}",
            )
            LOGGER.info("CAG cache created for %s (ttl=%dm).", shortcode, self.config.cache_ttl_minutes)
            return cache
        except Exception as exc:
            # Most common cause: context below the cache token minimum -> inline.
            LOGGER.info("Cache unavailable for %s (%s); using inline context.", shortcode, exc)
            return None

    def assess_comments_cag(
        self,
        prompt: str,
        cache: Optional[Any],
        inline_context_text: str = "",
        inline_frame_paths: Sequence[Path] = (),
    ) -> Optional[Dict[str, Any]]:
        """Score a comment chunk against the cached context (or inline fallback)."""
        if not self.available:
            return None
        cfg = GenerationConfig(
            temperature=self.config.llm_temperature,
            response_mime_type="application/json",
            response_schema=CAG_VIBE_RESPONSE_SCHEMA,
        )
        try:
            if cache is not None:
                model = GenerativeModel.from_cached_content(cached_content=cache)
                contents: List[Any] = [Part.from_text(prompt)]
                self.cache_hits += 1
            else:  # inline: resend context every call (no cache available)
                model = self._flash
                contents = [
                    Part.from_text(inline_context_text),
                    *self._image_parts(inline_frame_paths),
                    Part.from_text(prompt),
                ]
            self.call_count += 1
            resp = model.generate_content(contents, generation_config=cfg)
            return json.loads(resp.text)
        except Exception as exc:
            LOGGER.warning("CAG comment assessment failed: %s", exc)
            self.error_count += 1
            return None

    @staticmethod
    def release_cache(cache: Optional[Any]) -> None:
        if cache is None:
            return
        try:
            cache.delete()
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Cache delete failed: %s", exc)


# --------------------------------------------------------------------------- #
# Phase 1 — Multimodal Post Context Summarizer
# --------------------------------------------------------------------------- #
class MultimodalPostSummarizer:
    """Extract creative + visual context from frames + transcript + caption."""

    # Prompt text lives in prompts.py (single source of truth, easy to edit).
    _PROMPT = MULTIMODAL_POST_PROMPT

    def __init__(self, config: IGMultimodalConfig, client: VertexMultimodalClient) -> None:
        self.config = config
        self.client = client

    def _prompt(self, row: pd.Series) -> str:
        return self._PROMPT.format(
            subtype=row.get("media_subtype", "post"),
            caption=(str(row.get(CANON_POST_TEXT) or "")[:4000]) or "(nessuna didascalia)",
            transcript=(str(row.get(CANON_TRANSCRIPT) or "")[:8000]) or "(nessuna trascrizione)",
            formats=", ".join(ALLOWED_FORMATS),
            tones=", ".join(ALLOWED_TONES),
        )

    @staticmethod
    def _fallback(row: pd.Series) -> Dict[str, Any]:
        text = f"{row.get(CANON_POST_TEXT) or ''} {row.get(CANON_TRANSCRIPT) or ''}".lower()
        sponsored = bool(row.get("is_sponsored")) or any(
            k in text for k in ("#ad", "sponsor", "adv", "in collaborazione")
        )
        return {
            "format_type": "Sponsored Skit" if sponsored else "Organic Vlog",
            "primary_topic": "unknown",
            "intended_emotional_tone": "Neutral",
            "brand_entities": [],
            "visual_summary": "",
            "on_screen_text": [],
            "visual_setting": "",
        }

    def _validate(self, payload: Optional[Dict[str, Any]], row: pd.Series) -> Dict[str, Any]:
        if payload is None:
            return self._fallback(row)
        if _HAS_PYDANTIC:
            try:
                return MultimodalPostContext(**payload).model_dump()
            except ValidationError as exc:
                LOGGER.warning("MultimodalPostContext invalid (%s); fallback.", exc)
                return self._fallback(row)
        return {**self._fallback(row), **payload}

    def run(self, media_df: pd.DataFrame, posts: Sequence[IGPost]) -> pd.DataFrame:
        LOGGER.info("Phase 1 (multimodal): summarizing %d posts.", len(media_df))
        frames_by_code = {p.shortcode: p.frame_paths for p in posts}
        records: List[Dict[str, Any]] = []
        for _, row in media_df.iterrows():
            frames = frames_by_code.get(row[CANON_MEDIA_ID], [])
            payload = self.client.summarize_multimodal(self._prompt(row), frames)
            records.append(self._validate(payload, row))
        ctx = pd.DataFrame.from_records(records, index=media_df.index)
        return pd.concat([media_df, ctx], axis=1)


# --------------------------------------------------------------------------- #
# Phase 3 — Cache-Augmented community vibe over comments
# --------------------------------------------------------------------------- #
class CAGCommentAnalyzer:
    """Cache the post's multimodal context once, score comments in chunks."""

    def __init__(
        self,
        config: IGMultimodalConfig,
        client: VertexMultimodalClient,
        nlp: LocalNLPEnricher,
    ) -> None:
        self.config = config
        self.client = client
        self.nlp = nlp

    @staticmethod
    def _context_text(ctx: pd.Series) -> str:
        return (
            f"DIDASCALIA: {ctx.get(CANON_POST_TEXT) or ''}\n"
            f"TRASCRIZIONE: {ctx.get(CANON_TRANSCRIPT) or ''}\n"
            f"RIASSUNTO VISIVO: {ctx.get('visual_summary') or ''}\n"
            f"ARGOMENTO: {ctx.get('primary_topic') or ''} | TONO: {ctx.get('intended_emotional_tone') or ''}"
        )

    @staticmethod
    def _chunk_prompt(comments: Sequence[str]) -> str:
        body = "\n".join(f"- {str(c)[:300]}" for c in comments)
        return CAG_COMMENT_CHUNK_PROMPT.format(comments=body)

    def _validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if _HAS_PYDANTIC:
            try:
                return CAGCommunityVibe(**payload).model_dump()
            except ValidationError:
                pass
        return {
            "sentiment_polarization_index": float(payload.get("sentiment_polarization_index", 0.5)),
            "dominant_community_emotion": str(payload.get("dominant_community_emotion", "mixed")),
            "community_noun_phrases": list(payload.get("community_noun_phrases", []) or []),
            "visual_reference_ratio": float(payload.get("visual_reference_ratio", 0.0)),
        }

    def _chunks(self, texts: List[str]) -> List[List[str]]:
        k = self.config.comment_chunk_size
        return [texts[i : i + k] for i in range(0, len(texts), k)] or [[]]

    def run(
        self,
        enriched_comments: pd.DataFrame,
        post_context: pd.DataFrame,
        posts: Sequence[IGPost],
        raw_comments: pd.DataFrame,
    ) -> pd.DataFrame:
        LOGGER.info("Phase 3 (CAG): %d posts.", post_context[CANON_MEDIA_ID].nunique())
        frames_by_code = {p.shortcode: p.frame_paths for p in posts}
        ctx_indexed = post_context.set_index(CANON_MEDIA_ID)

        # Deterministic structural aggregates (vectorized).
        if not enriched_comments.empty:
            agg = enriched_comments.groupby(CANON_MEDIA_ID).agg(
                analyzed_comment_count=(CANON_COMMENT_ID, "count"),
                mean_cosine_similarity=("context_cosine_similarity", "mean"),
                mean_entity_overlap=("entity_overlap_count", "mean"),
                mean_emoji_density=("emoji_density", "mean"),
                mean_noun_jaccard=("noun_jaccard_vs_post", "mean"),
            )
        else:
            agg = pd.DataFrame()

        # Enriched groups carry NLP features (cosine, _noun_set); raw groups carry text.
        enriched_by_media = (
            dict(tuple(enriched_comments.groupby(CANON_MEDIA_ID)))
            if not enriched_comments.empty else {}
        )
        raw_by_media = (
            dict(tuple(raw_comments.groupby(CANON_MEDIA_ID)))
            if not raw_comments.empty and CANON_MEDIA_ID in raw_comments else {}
        )

        records: List[Dict[str, Any]] = []
        for media_id in ctx_indexed.index:
            ctx = ctx_indexed.loc[media_id]
            sub = enriched_by_media.get(media_id)
            raw_sub = raw_by_media.get(media_id)
            texts = (
                raw_sub[CANON_COMMENT_TEXT].dropna().astype(str).tolist()
                if raw_sub is not None else []
            )

            # Local structural baselines (always available).
            post_nouns = self.nlp.noun_set(str(ctx.get(CANON_POST_TEXT) or ""))
            comm_nouns: set[str] = set()
            if sub is not None and "_noun_set" in sub.columns:
                for s in sub["_noun_set"]:
                    comm_nouns |= set(s)
            topical_adherence = self.nlp.jaccard(comm_nouns, post_nouns)
            polarization = self._local_polarization(sub)
            dominant_emotion, visual_ref = "mixed", 0.0
            community_phrases = sorted(comm_nouns)[:15]

            # --- CAG path -----------------------------------------------------
            if texts and len(texts) >= self.config.min_comments_for_vibe and self.client.available:
                frames = frames_by_code.get(media_id, [])
                ctx_text = self._context_text(ctx)
                cache = self.client.build_cache(str(media_id), ctx_text, frames)
                chunk_results: List[Dict[str, Any]] = []
                sampled = texts[: self.config.max_comments_sampled]
                for chunk in self._chunks(sampled):
                    if not chunk:
                        continue
                    payload = self.client.assess_comments_cag(
                        self._chunk_prompt(chunk), cache,
                        inline_context_text=ctx_text, inline_frame_paths=frames,
                    )
                    if payload is not None:
                        chunk_results.append(self._validate(payload))
                self.client.release_cache(cache)

                if chunk_results:
                    polarization = float(np.mean([r["sentiment_polarization_index"] for r in chunk_results]))
                    visual_ref = float(np.mean([r["visual_reference_ratio"] for r in chunk_results]))
                    dominant_emotion = chunk_results[0]["dominant_community_emotion"]
                    phrases: set[str] = set()
                    for r in chunk_results:
                        phrases |= {p.lower() for p in r["community_noun_phrases"]}
                    if phrases:
                        community_phrases = sorted(phrases)[:20]
                        topical_adherence = max(topical_adherence, self.nlp.jaccard(phrases, post_nouns))

            record = {
                CANON_MEDIA_ID: media_id,
                "sentiment_polarization_index": polarization,
                "topical_adherence_score": topical_adherence,
                "visual_reference_ratio": visual_ref,
                "dominant_community_emotion": dominant_emotion,
                "community_noun_phrases": community_phrases,
            }
            if not agg.empty and media_id in agg.index:
                record.update(agg.loc[media_id].to_dict())
            records.append(record)

        return pd.DataFrame.from_records(records)

    @staticmethod
    def _local_polarization(sub: Optional[pd.DataFrame]) -> float:
        if sub is None or "context_cosine_similarity" not in (sub.columns if sub is not None else []):
            return 0.5
        sims = sub["context_cosine_similarity"].to_numpy(dtype=float)
        if sims.size < 2:
            return 0.5
        return float(np.clip(np.std(sims) / 0.5, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class InstagramMultimodalPipeline:
    """discover → Phase 1 (multimodal) → Phase 2 (local NLP) → Phase 3 (CAG)."""

    def __init__(self, config: Optional[IGMultimodalConfig] = None) -> None:
        self.config = config or IGMultimodalConfig()
        self.loader = IGMultimodalLoader(self.config)
        self.client = VertexMultimodalClient(self.config)
        self.nlp = LocalNLPEnricher(self.config)
        self.summarizer = MultimodalPostSummarizer(self.config, self.client)
        self.analyzer = CAGCommentAnalyzer(self.config, self.client, self.nlp)

    # Extract IG shortcode from a permalink: /p/<code>/, /reel/<code>/, /tv/<code>/
    _SHORTCODE_RE = __import__("re").compile(r"/(?:p|reel|tv)/([^/?#]+)")

    def run(self, external_comments: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        LOGGER.info("=== IG multimodal pipeline START ===")
        posts = self.loader.discover()
        if not posts:
            raise ValueError("No Instagram posts discovered under configured roots.")
        media_df, embedded_comments = self.loader.to_frames(posts)

        # Backfill numeric media_id + engagement features from ig_posts_cleaned,
        # using the permalink→shortcode bridge (lets reels join the corpus too).
        media_df = self._merge_posts_metadata(media_df)

        # Comment source priority: explicit arg → ig_comments_cleaned → embedded.
        if external_comments is None and self.config.ig_comments_parquet:
            external_comments = self._load_parquet(self.config.ig_comments_parquet)
        comments = self._select_comments(external_comments, embedded_comments, media_df)
        LOGGER.info("Media: %d | comments for analysis: %d", len(media_df), len(comments))

        # --- Phase 1 (multimodal) -------------------------------------------
        post_context = self.summarizer.run(media_df, posts)

        # --- Phase 2 (reused local NLP) -------------------------------------
        if comments.empty:
            enriched = pd.DataFrame(columns=[CANON_MEDIA_ID, CANON_COMMENT_ID])
        else:
            enriched = self.nlp.enrich_comments(comments, post_context)

        # --- Phase 3 (CAG) --------------------------------------------------
        vibe = self.analyzer.run(enriched, post_context, posts, comments)

        matrix = post_context.merge(vibe, on=CANON_MEDIA_ID, how="left", suffixes=("", "_vibe"))
        matrix = matrix.drop(columns=[CANON_TRANSCRIPT], errors="ignore")
        self._persist(matrix)
        LOGGER.info(
            "=== DONE === posts=%d | LLM calls=%d cache_hits=%d errors=%d",
            len(matrix), self.client.call_count, self.client.cache_hits, self.client.error_count,
        )
        return matrix

    @staticmethod
    def _load_parquet(path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(path)
            LOGGER.info("Loaded %s (%d rows).", path, len(df))
            return df
        except Exception as exc:
            LOGGER.warning("Could not load %s: %s", path, exc)
            return None

    def _merge_posts_metadata(self, media_df: pd.DataFrame) -> pd.DataFrame:
        """Backfill numeric media_id + engagement columns from ig_posts_cleaned."""
        if not self.config.ig_posts_parquet:
            return media_df
        posts = self._load_parquet(self.config.ig_posts_parquet)
        if posts is None or "permalink" not in posts.columns:
            return media_df

        posts = posts.copy()
        posts["shortcode"] = posts["permalink"].astype(str).str.extract(self._SHORTCODE_RE)
        engagement = [c for c in ("media_type", "media_product_type", "reach", "views",
                                  "total_interactions", "saved") if c in posts.columns]
        meta = posts.dropna(subset=["shortcode"]).set_index("shortcode")

        # The cleaned posts/comments corpora share one id space, which differs
        # from the instaloader node.id. So the ig_posts_cleaned media_id (reached
        # via shortcode) is AUTHORITATIVE for joining the comment corpus.
        sc_to_id = (
            meta[~meta.index.duplicated()]["media_id"].astype(str).to_dict()
            if "media_id" in meta else {}
        )
        out = media_df.copy()
        out["media_id_numeric"] = out.apply(
            lambda r: sc_to_id.get(r[CANON_MEDIA_ID]) or r["media_id_numeric"], axis=1
        )
        if engagement:
            out = out.merge(
                meta[engagement], left_on=CANON_MEDIA_ID, right_index=True, how="left"
            )
        matched = out["media_id_numeric"].notna().sum()
        LOGGER.info("Posts metadata merged: %d/%d posts now have numeric media_id.",
                    matched, len(out))
        return out

    def _select_comments(
        self,
        external: Optional[pd.DataFrame],
        embedded: pd.DataFrame,
        media_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if external is None or external.empty:
            return embedded
        # Normalize raw platform columns (text→comment_text, from_id→author_id, …).
        ext = SchemaNormalizer.normalize_comments(external, "instagram")
        # Remap numeric media_id → shortcode (the media frame's canonical key).
        id_map = (
            media_df.dropna(subset=["media_id_numeric"])
            .assign(_num=lambda d: d["media_id_numeric"].astype(str))
            .set_index("_num")[CANON_MEDIA_ID]
            .to_dict()
        )
        ext[CANON_MEDIA_ID] = (
            ext[CANON_MEDIA_ID].astype(str).map(id_map).fillna(ext[CANON_MEDIA_ID])
        )
        keep = set(media_df[CANON_MEDIA_ID])
        ext = ext[ext[CANON_MEDIA_ID].isin(keep)]
        LOGGER.info("Using external comment corpus: %d rows matched to %d posts.",
                    len(ext), ext[CANON_MEDIA_ID].nunique())
        return ext if len(ext) else embedded

    def _persist(self, matrix: pd.DataFrame) -> None:
        out = str(self.config.output_path)
        df = matrix.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, tuple, set))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(list(x), ensure_ascii=False)
                    if isinstance(x, (list, tuple, set)) else x
                )
        try:
            df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
            LOGGER.info("Wrote IG multimodal matrix → %s", out)
        except Exception as exc:
            csv_out = (out[:-8] + ".csv") if out.endswith(".parquet") else out + ".csv"
            LOGGER.error("Parquet write failed (%s); CSV → %s", exc, csv_out)
            df.to_csv(csv_out, index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # enable_llm=False → runs locally on the real folders with deterministic
    # fallbacks (no Vertex). Set True on Colab Enterprise for multimodal + CAG.
    config = IGMultimodalConfig(
        enable_llm=False,
        max_posts=8,                      # smoke run; set None for the full set
        output_path="enriched_ig_multimodal_vibe_matrix.parquet",
    )
    pipeline = InstagramMultimodalPipeline(config)
    matrix = pipeline.run()

    pd.set_option("display.max_columns", None, "display.width", 240)
    cols = [c for c in (CANON_MEDIA_ID, "media_subtype", "n_frames", "format_type",
                        "primary_topic", "sentiment_polarization_index",
                        "topical_adherence_score", "visual_reference_ratio",
                        "comment_count") if c in matrix.columns]
    LOGGER.info("Result preview:\n%s", matrix[cols].to_string(index=False))


if __name__ == "__main__":
    main()
