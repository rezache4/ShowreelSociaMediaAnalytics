"""Show Reel Media Group — Post-Level Context Enrichment & Community Vibe Baseline.

This module implements Step 1 (Post-Level Context Enrichment) and Step 2
(Community Vibe Baseline) of the cross-platform audience-dynamics framework.

It ingests platform-specific media + interaction tables (Instagram, Facebook,
TikTok, plus YouTube transcripts), normalizes them to a canonical schema, and
produces a single, analysis-ready ``enriched_post_vibe_matrix.parquet`` that
fuses three signal layers:

    Phase 1 — Post-Level Context Summarizer   (Gemini 2.5 Flash, high-throughput)
        format_type / primary_topic / intended_emotional_tone / brand_entities

    Phase 2 — High-Velocity Local NLP Enrichment   (spaCy it_core_news_lg)
        token metrics, comment↔post entity overlap, cosine vector similarity

    Phase 3 — Community Vibe & Polarization Aggregator   (Gemini 2.5 Pro)
        sentiment_polarization_index + Jaccard topical_adherence_score per media

Design notes
------------
* Object-oriented, fully type-hinted, ``logging``-based, defensive try/except.
* LLM calls return *strict* JSON (controlled generation + Pydantic validation),
  never markdown-wrapped payloads.
* Heavy dependencies (vertexai, spaCy, emoji) are import-guarded so the bundled
  mock ``main`` block runs end-to-end even on a bare interpreter, degrading to
  deterministic local stubs and logging the downgrade loudly.

Author: AFB_Lab — Lead Data Science / MLOps
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prompts import COMMUNITY_VIBE_PROMPT, POST_CONTEXT_PROMPT

# --------------------------------------------------------------------------- #
# Optional heavy dependencies — import-guarded for graceful degradation.
# --------------------------------------------------------------------------- #
try:  # Probabilistic LLM layer
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    _HAS_VERTEX = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_VERTEX = False

try:  # Deterministic NLP layer
    import spacy

    _HAS_SPACY = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_SPACY = False

try:  # ZWJ-safe emoji extraction
    import emoji as emoji_lib

    _HAS_EMOJI = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_EMOJI = False

try:  # Structured-output validation
    from pydantic import BaseModel, Field, ValidationError, field_validator

    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PYDANTIC = False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("community_vibe_pipeline")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig:
    """Centralized, immutable configuration for the enrichment pipeline."""

    # --- Vertex AI / Gemini -------------------------------------------------
    gcp_project_id: str = "gen-lang-client-0792749758"
    gcp_location: str = "us-central1"
    flash_model: str = "gemini-2.5-flash"  # Phase 1: high-throughput summaries
    pro_model: str = "gemini-2.5-pro"      # Phase 3: nuanced community baselines

    # --- spaCy --------------------------------------------------------------
    spacy_model: str = "it_core_news_lg"

    # --- Throughput / sampling ---------------------------------------------
    post_batch_size: int = 32              # posts per Flash batch
    max_comments_sampled: int = 60         # comments fed to Pro per media
    min_comments_for_vibe: int = 3         # below this, skip LLM vibe scoring
    llm_max_retries: int = 3
    llm_temperature: float = 0.1

    # --- IO -----------------------------------------------------------------
    output_path: Path = Path("enriched_post_vibe_matrix.parquet")

    # --- LLM toggle (disable to run a fully local/offline pass) ------------
    enable_llm: bool = True


# Canonical schema. Every platform table is mapped onto these column names.
CANON_MEDIA_ID = "media_id"
CANON_POST_TEXT = "post_text"
CANON_TRANSCRIPT = "transcript"
CANON_PLATFORM = "platform"
CANON_COMMENT_ID = "comment_id"
CANON_PARENT_ID = "parent_id"
CANON_AUTHOR_ID = "author_id"
CANON_COMMENT_TEXT = "comment_text"
CANON_TIMESTAMP = "timestamp"


# Per-platform raw→canonical column maps. Multiple candidate source names are
# tried in order; the first present column wins. This accommodates the documented
# IG/FB (`from_id`, `media_id`, `text`, `parent_id`) vs TikTok (`uid`,
# `video_id`, `reply_id`) divergence plus FB's `message` body field.
MEDIA_COLUMN_MAP: Dict[str, Dict[str, Sequence[str]]] = {
    "instagram": {
        CANON_MEDIA_ID: ("media_id",),
        CANON_POST_TEXT: ("caption", "message"),
        CANON_TRANSCRIPT: ("transcript",),
        CANON_TIMESTAMP: ("timestamp",),
    },
    "facebook": {
        CANON_MEDIA_ID: ("post_id", "media_id"),
        CANON_POST_TEXT: ("message", "caption"),
        CANON_TRANSCRIPT: ("transcript",),
        CANON_TIMESTAMP: ("timestamp",),
    },
    "tiktok": {
        CANON_MEDIA_ID: ("media_id", "video_id", "post_id"),
        CANON_POST_TEXT: ("caption", "description", "message"),
        CANON_TRANSCRIPT: ("transcript",),
        CANON_TIMESTAMP: ("timestamp",),
    },
    "youtube": {
        CANON_MEDIA_ID: ("video_id", "media_id"),
        CANON_POST_TEXT: ("title", "description"),
        CANON_TRANSCRIPT: ("transcript", "captions"),
        CANON_TIMESTAMP: ("timestamp", "published_at"),
    },
}

COMMENT_COLUMN_MAP: Dict[str, Dict[str, Sequence[str]]] = {
    "instagram": {
        CANON_COMMENT_ID: ("comment_id",),
        CANON_MEDIA_ID: ("media_id",),
        CANON_PARENT_ID: ("parent_id", "reply_to_comment_id"),
        CANON_AUTHOR_ID: ("from_id", "author_id"),
        CANON_COMMENT_TEXT: ("text", "message"),
        CANON_TIMESTAMP: ("timestamp",),
    },
    "facebook": {
        CANON_COMMENT_ID: ("comment_id",),
        CANON_MEDIA_ID: ("post_id", "media_id"),
        CANON_PARENT_ID: ("parent_id", "reply_to_comment_id"),
        CANON_AUTHOR_ID: ("from_id", "author_id"),
        CANON_COMMENT_TEXT: ("message", "text"),
        CANON_TIMESTAMP: ("timestamp",),
    },
    "tiktok": {
        CANON_COMMENT_ID: ("comment_id",),
        CANON_MEDIA_ID: ("media_id", "video_id"),
        CANON_PARENT_ID: ("reply_id", "parent_id"),
        CANON_AUTHOR_ID: ("uid", "from_id", "author_id"),
        CANON_COMMENT_TEXT: ("text", "message"),
        CANON_TIMESTAMP: ("timestamp",),
    },
}


# --------------------------------------------------------------------------- #
# Structured-output schemas (Pydantic + raw Vertex response schemas)
# --------------------------------------------------------------------------- #
ALLOWED_FORMATS = (
    "Sponsored Skit",
    "Organic Vlog",
    "Product Review",
    "Tutorial",
    "Q&A",
    "Behind The Scenes",
    "News Commentary",
    "Meme",
    "Announcement",
    "Other",
)
ALLOWED_TONES = (
    "Humorous",
    "Inspirational",
    "Informative",
    "Nostalgic",
    "Provocative",
    "Heartfelt",
    "Neutral",
    "Promotional",
)


if _HAS_PYDANTIC:

    class PostContext(BaseModel):
        """Phase 1 deterministic contract for a single post summary."""

        format_type: str = Field(..., description="Content format archetype.")
        primary_topic: str = Field(..., description="Concise main subject (<=6 words).")
        intended_emotional_tone: str = Field(..., description="Dominant authored tone.")
        brand_entities: List[str] = Field(
            default_factory=list, description="Explicit corporate / brand mentions."
        )

        @field_validator("brand_entities", mode="before")
        @classmethod
        def _coerce_list(cls, v: Any) -> List[str]:
            if v is None:
                return []
            if isinstance(v, str):
                return [v] if v.strip() else []
            return list(v)

    class CommunityVibe(BaseModel):
        """Phase 3 contract for the aggregated community baseline of a media."""

        sentiment_polarization_index: float = Field(
            ..., ge=0.0, le=1.0,
            description="0.0 = full consensus, 1.0 = fractured/polarized.",
        )
        dominant_community_emotion: str = Field(
            ..., description="Modal audience reaction."
        )
        community_noun_phrases: List[str] = Field(
            default_factory=list,
            description="Salient noun phrases the audience converges on.",
        )

        @field_validator("sentiment_polarization_index", mode="before")
        @classmethod
        def _clamp(cls, v: Any) -> float:
            try:
                return float(min(1.0, max(0.0, float(v))))
            except (TypeError, ValueError):
                return 0.5

        @field_validator("community_noun_phrases", mode="before")
        @classmethod
        def _coerce_list(cls, v: Any) -> List[str]:
            if v is None:
                return []
            if isinstance(v, str):
                return [v] if v.strip() else []
            return list(v)
else:  # pragma: no cover - pydantic-free fallback
    PostContext = dict  # type: ignore
    CommunityVibe = dict  # type: ignore


# Vertex controlled-generation response schemas (OpenAPI subset). These force the
# model to emit raw JSON matching the Pydantic contracts above — no markdown.
POST_CONTEXT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "format_type": {"type": "STRING", "enum": list(ALLOWED_FORMATS)},
        "primary_topic": {"type": "STRING"},
        "intended_emotional_tone": {"type": "STRING", "enum": list(ALLOWED_TONES)},
        "brand_entities": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["format_type", "primary_topic", "intended_emotional_tone", "brand_entities"],
}

COMMUNITY_VIBE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "sentiment_polarization_index": {"type": "NUMBER"},
        "dominant_community_emotion": {"type": "STRING"},
        "community_noun_phrases": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "sentiment_polarization_index",
        "dominant_community_emotion",
        "community_noun_phrases",
    ],
}


# --------------------------------------------------------------------------- #
# Schema normalization
# --------------------------------------------------------------------------- #
class SchemaNormalizer:
    """Maps heterogeneous platform tables onto the canonical schema.

    Performance: applies strict dtype downcasting and converts identifier and
    text columns to memory-frugal ``string`` dtype to keep large cross-sectional
    frames lean.
    """

    @staticmethod
    def _resolve(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    @classmethod
    def _normalize(
        cls,
        df: pd.DataFrame,
        platform: str,
        column_map: Dict[str, Dict[str, Sequence[str]]],
        required: Sequence[str],
    ) -> pd.DataFrame:
        if platform not in column_map:
            raise KeyError(f"Unsupported platform '{platform}'. Known: {list(column_map)}")

        mapping = column_map[platform]
        out = pd.DataFrame(index=df.index)
        for canon, candidates in mapping.items():
            source = cls._resolve(df, candidates)
            if source is not None:
                out[canon] = df[source]
            else:
                out[canon] = pd.NA
        out[CANON_PLATFORM] = platform

        missing = [c for c in required if out[c].isna().all()]
        if missing:
            LOGGER.warning(
                "[%s] required canonical columns absent in source: %s", platform, missing
            )

        # Identifiers + text → string dtype (nullable, vectorized, low overhead).
        for col in out.columns:
            if out[col].dtype == object or col.endswith("_id") or "text" in col:
                out[col] = out[col].astype("string")
        return out

    @classmethod
    def normalize_media(cls, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        norm = cls._normalize(
            df, platform, MEDIA_COLUMN_MAP, required=[CANON_MEDIA_ID, CANON_POST_TEXT]
        )
        # Drop rows lacking a usable media id.
        before = len(norm)
        norm = norm.dropna(subset=[CANON_MEDIA_ID]).reset_index(drop=True)
        LOGGER.info(
            "[%s] media normalized: %d → %d rows (dropped %d id-less)",
            platform, before, len(norm), before - len(norm),
        )
        return norm

    @classmethod
    def normalize_comments(cls, df: pd.DataFrame, platform: str) -> pd.DataFrame:
        norm = cls._normalize(
            df, platform, COMMENT_COLUMN_MAP,
            required=[CANON_MEDIA_ID, CANON_COMMENT_TEXT],
        )
        before = len(norm)
        norm = norm.dropna(subset=[CANON_MEDIA_ID, CANON_COMMENT_TEXT]).reset_index(drop=True)
        LOGGER.info(
            "[%s] comments normalized: %d → %d rows (dropped %d incomplete)",
            platform, before, len(norm), before - len(norm),
        )
        return norm


# --------------------------------------------------------------------------- #
# Vertex AI client wrapper
# --------------------------------------------------------------------------- #
class VertexLLMClient:
    """Thin, retrying wrapper around Vertex AI Gemini with controlled generation.

    Returns parsed ``dict`` payloads. On any failure (SDK missing, transport
    error, malformed JSON) it logs and returns ``None`` so callers can fall back
    to deterministic defaults — individual failures never halt the batch.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._flash: Optional["GenerativeModel"] = None
        self._pro: Optional["GenerativeModel"] = None
        self.available = False
        self.call_count = 0
        self.error_count = 0

        if not (config.enable_llm and _HAS_VERTEX):
            LOGGER.warning(
                "Vertex AI disabled or unavailable (enable_llm=%s, sdk=%s). "
                "LLM phases will use deterministic fallbacks.",
                config.enable_llm, _HAS_VERTEX,
            )
            return
        try:
            vertexai.init(project=config.gcp_project_id, location=config.gcp_location)
            self._flash = GenerativeModel(config.flash_model)
            self._pro = GenerativeModel(config.pro_model)
            self.available = True
            LOGGER.info(
                "Vertex AI initialized (project=%s, location=%s).",
                config.gcp_project_id, config.gcp_location,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.error("Vertex AI init failed: %s", exc)

    def _generate(
        self,
        model: "GenerativeModel",
        prompt: str,
        response_schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        gen_config = GenerationConfig(
            temperature=self.config.llm_temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        for attempt in range(1, self.config.llm_max_retries + 1):
            try:
                self.call_count += 1
                resp = model.generate_content(prompt, generation_config=gen_config)
                return json.loads(resp.text)
            except json.JSONDecodeError as exc:
                LOGGER.warning("JSON decode failure (attempt %d): %s", attempt, exc)
            except Exception as exc:  # transport / quota / safety
                LOGGER.warning("Gemini call failed (attempt %d): %s", attempt, exc)
        self.error_count += 1
        return None

    def summarize_post(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.available or self._flash is None:
            return None
        return self._generate(self._flash, prompt, POST_CONTEXT_RESPONSE_SCHEMA)

    def assess_community(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.available or self._pro is None:
            return None
        return self._generate(self._pro, prompt, COMMUNITY_VIBE_RESPONSE_SCHEMA)


# --------------------------------------------------------------------------- #
# Phase 1 — Post-Level Context Summarizer
# --------------------------------------------------------------------------- #
class PostContextSummarizer:
    """Phase 1: batch posts and extract structured creative context via Flash."""

    # Prompt text lives in prompts.py (single source of truth, easy to edit).
    _PROMPT_TEMPLATE = POST_CONTEXT_PROMPT

    def __init__(self, config: PipelineConfig, llm: VertexLLMClient) -> None:
        self.config = config
        self.llm = llm

    def _build_prompt(self, row: pd.Series) -> str:
        transcript = row.get(CANON_TRANSCRIPT)
        transcript_txt = "" if pd.isna(transcript) else str(transcript)[:8000]
        post_text = row.get(CANON_POST_TEXT)
        post_txt = "" if pd.isna(post_text) else str(post_text)[:4000]
        return self._PROMPT_TEMPLATE.format(
            platform=row.get(CANON_PLATFORM, "social"),
            post_text=post_txt or "(nessuna didascalia)",
            transcript=transcript_txt or "(nessuna trascrizione)",
            formats=", ".join(ALLOWED_FORMATS),
            tones=", ".join(ALLOWED_TONES),
        )

    @staticmethod
    def _fallback(row: pd.Series) -> Dict[str, Any]:
        """Deterministic, dependency-free context when the LLM is unavailable."""
        text = " ".join(
            str(row.get(c, "") or "") for c in (CANON_POST_TEXT, CANON_TRANSCRIPT)
        ).lower()
        sponsored = any(k in text for k in ("#ad", "sponsor", "adv", "in collaborazione"))
        return {
            "format_type": "Sponsored Skit" if sponsored else "Organic Vlog",
            "primary_topic": "unknown",
            "intended_emotional_tone": "Neutral",
            "brand_entities": [],
        }

    def _validate(self, payload: Optional[Dict[str, Any]], row: pd.Series) -> Dict[str, Any]:
        if payload is None:
            return self._fallback(row)
        if _HAS_PYDANTIC:
            try:
                return PostContext(**payload).model_dump()
            except ValidationError as exc:
                LOGGER.warning("PostContext validation failed (%s); using fallback.", exc)
                return self._fallback(row)
        return {**self._fallback(row), **payload}

    def run(self, media_df: pd.DataFrame) -> pd.DataFrame:
        """Return ``media_df`` augmented with Phase-1 context columns."""
        LOGGER.info("Phase 1: summarizing %d posts (batch=%d).",
                    len(media_df), self.config.post_batch_size)
        records: List[Dict[str, Any]] = []
        for start in range(0, len(media_df), self.config.post_batch_size):
            batch = media_df.iloc[start : start + self.config.post_batch_size]
            for _, row in batch.iterrows():
                payload = self.llm.summarize_post(self._build_prompt(row))
                records.append(self._validate(payload, row))
            LOGGER.info("Phase 1: processed %d/%d posts.",
                        min(start + self.config.post_batch_size, len(media_df)), len(media_df))

        ctx = pd.DataFrame.from_records(records, index=media_df.index)
        return pd.concat([media_df, ctx], axis=1)


# --------------------------------------------------------------------------- #
# Phase 2 — High-Velocity Local NLP Enrichment
# --------------------------------------------------------------------------- #
_EMOJI_REGEX = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF"
    "\U00002700-\U000027BF\U0001F900-\U0001F9FF]+"
)
_PUNCT_REGEX = re.compile(r"[!?.,;:…]")
_WORD_REGEX = re.compile(r"\b\w+\b", re.UNICODE)


class LocalNLPEnricher:
    """Phase 2: deterministic spaCy-driven enrichment at high velocity.

    Computes token-level metrics, exact comment-noun↔post-entity overlap, and the
    cosine similarity between comment and post-context vectors:

        Similarity(u, v) = (u · v) / (||u|| * ||v||)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.nlp = self._load_model(config.spacy_model)
        self.has_vectors = bool(self.nlp) and self.nlp.vocab.vectors_length > 0  # type: ignore
        if self.nlp is None:
            LOGGER.warning(
                "spaCy model '%s' unavailable; Phase 2 uses lexical fallbacks "
                "(zero vectors).", config.spacy_model,
            )

    @staticmethod
    def _load_model(name: str) -> Optional["spacy.language.Language"]:
        if not _HAS_SPACY:
            return None
        try:
            return spacy.load(name)
        except Exception as exc:  # pragma: no cover - model not downloaded
            LOGGER.error("Failed to load spaCy model '%s': %s", name, exc)
            return None

    # --- token-level metrics -------------------------------------------------
    @staticmethod
    def _extract_emojis(text: str) -> List[str]:
        if _HAS_EMOJI:
            return [e["emoji"] for e in emoji_lib.emoji_list(text)]
        return _EMOJI_REGEX.findall(text)

    @classmethod
    def token_metrics(cls, text: str) -> Dict[str, float]:
        text = text or ""
        words = _WORD_REGEX.findall(text)
        word_count = len(words)
        emojis = cls._extract_emojis(text)
        emoji_count = len(emojis)
        punct_count = len(_PUNCT_REGEX.findall(text))
        char_count = max(len(text), 1)
        return {
            "word_count": float(word_count),
            "char_count": float(len(text)),
            "emoji_count": float(emoji_count),
            # density: emojis relative to lexical payload (per word, +1 smoothed)
            "emoji_density": float(emoji_count / (word_count + 1)),
            # velocity: punctuation marks per character (intensity of cadence)
            "punctuation_velocity": float(punct_count / char_count),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        }

    # --- vectors & nouns -----------------------------------------------------
    def _doc(self, text: str) -> Optional["spacy.tokens.Doc"]:
        if self.nlp is None or not text:
            return None
        try:
            return self.nlp(text[:10000])
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("spaCy parse failed: %s", exc)
            return None

    def vector(self, text: str) -> np.ndarray:
        doc = self._doc(text)
        if doc is not None and self.has_vectors and doc.has_vector:
            return np.asarray(doc.vector, dtype=np.float32)
        return np.zeros(self.nlp.vocab.vectors_length if self.has_vectors else 1, dtype=np.float32)  # type: ignore

    def noun_set(self, text: str) -> set[str]:
        """Lemmatized noun + noun-chunk-head set for overlap/Jaccard."""
        doc = self._doc(text)
        if doc is None:
            return {w.lower() for w in _WORD_REGEX.findall(text or "") if len(w) > 3}
        nouns = {t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop}
        nouns |= {chunk.root.lemma_.lower() for chunk in doc.noun_chunks}
        return {n for n in nouns if n.strip()}

    def entity_set(self, text: str) -> set[str]:
        """Named entities (lowercased) used as the post's anchor entity array."""
        doc = self._doc(text)
        if doc is None:
            return set()
        return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

    @staticmethod
    def jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        return float(len(a & b) / len(union)) if union else 0.0

    def enrich_comments(
        self, comments_df: pd.DataFrame, post_context: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach per-comment NLP features keyed against the parent post context.

        ``post_context`` must be indexed/identified by ``media_id`` and carry the
        Phase-1 fields (``brand_entities``, ``primary_topic``) plus ``post_text``.
        """
        LOGGER.info("Phase 2: enriching %d comments via spaCy.", len(comments_df))

        # Pre-compute post-side artifacts once per media (avoids re-parsing).
        post_text_col = post_context[CANON_POST_TEXT].fillna("")
        post_ctx_text = (
            post_text_col
            + " "
            + post_context.get("primary_topic", pd.Series("", index=post_context.index)).fillna("")
        )
        brand_col = post_context.get(
            "brand_entities", pd.Series([[]] * len(post_context), index=post_context.index)
        )
        post_vectors: Dict[str, np.ndarray] = {}
        post_entities: Dict[str, set[str]] = {}
        post_nouns: Dict[str, set[str]] = {}
        for media_id, ctx_text, body_text, brands in zip(
            post_context[CANON_MEDIA_ID], post_ctx_text, post_text_col, brand_col
        ):
            mid = str(media_id)
            post_vectors[mid] = self.vector(ctx_text)
            ents = self.entity_set(str(body_text))
            if isinstance(brands, (list, tuple)):
                ents |= {str(b).lower() for b in brands}
            post_entities[mid] = ents
            post_nouns[mid] = self.noun_set(ctx_text)

        out_rows: List[Dict[str, Any]] = []
        for _, row in comments_df.iterrows():
            text = "" if pd.isna(row[CANON_COMMENT_TEXT]) else str(row[CANON_COMMENT_TEXT])
            mid = str(row[CANON_MEDIA_ID])
            metrics = self.token_metrics(text)

            c_vec = self.vector(text)
            c_nouns = self.noun_set(text)
            p_vec = post_vectors.get(mid, np.zeros_like(c_vec))
            p_ents = post_entities.get(mid, set())
            p_nouns = post_nouns.get(mid, set())

            entity_overlap = len(c_nouns & p_ents)
            metrics.update(
                {
                    CANON_COMMENT_ID: row.get(CANON_COMMENT_ID),
                    CANON_MEDIA_ID: mid,
                    CANON_PLATFORM: row.get(CANON_PLATFORM),
                    "entity_overlap_count": float(entity_overlap),
                    "entity_overlap_ratio": float(entity_overlap / (len(c_nouns) or 1)),
                    "context_cosine_similarity": self.cosine_similarity(c_vec, p_vec),
                    "noun_jaccard_vs_post": self.jaccard(c_nouns, p_nouns),
                    "_noun_set": sorted(c_nouns),  # retained for Phase-3 aggregation
                }
            )
            out_rows.append(metrics)

        enriched = pd.DataFrame.from_records(out_rows)
        # Strict downcast of float feature columns.
        float_cols = enriched.select_dtypes(include="float").columns
        enriched[float_cols] = enriched[float_cols].apply(pd.to_numeric, downcast="float")
        LOGGER.info("Phase 2: produced %d enriched comment rows.", len(enriched))
        return enriched


# --------------------------------------------------------------------------- #
# Phase 3 — Community Vibe & Polarization Aggregator
# --------------------------------------------------------------------------- #
class CommunityVibeAggregator:
    """Phase 3: per-media community baseline via group-by + Gemini 2.5 Pro."""

    # Prompt text lives in prompts.py (single source of truth, easy to edit).
    _PROMPT_TEMPLATE = COMMUNITY_VIBE_PROMPT

    def __init__(
        self,
        config: PipelineConfig,
        llm: VertexLLMClient,
        nlp: LocalNLPEnricher,
    ) -> None:
        self.config = config
        self.llm = llm
        self.nlp = nlp

    @staticmethod
    def _sample_comments(group: pd.DataFrame, k: int) -> pd.DataFrame:
        """Bias the sample toward top-level comments, then fill with replies."""
        if len(group) <= k:
            return group
        is_reply = group[CANON_PARENT_ID].notna() if CANON_PARENT_ID in group else pd.Series(False, index=group.index)
        top = group[~is_reply]
        replies = group[is_reply]
        n_top = min(len(top), max(k // 2, k - len(replies)))
        n_rep = k - n_top
        parts = [top.sample(n=n_top, random_state=42) if n_top else top.head(0),
                 replies.sample(n=min(n_rep, len(replies)), random_state=42) if n_rep else replies.head(0)]
        return pd.concat(parts)

    def _build_prompt(self, sample: pd.DataFrame, context: pd.Series) -> str:
        comment_lines = "\n".join(
            f"- {str(t)[:300]}" for t in sample[CANON_COMMENT_TEXT].dropna().tolist()
        )
        return self._PROMPT_TEMPLATE.format(
            platform=context.get(CANON_PLATFORM, "social"),
            topic=context.get("primary_topic", "n/a"),
            tone=context.get("intended_emotional_tone", "n/a"),
            comments=comment_lines or "(nessun commento)",
        )

    @staticmethod
    def _polarization_fallback(metrics: pd.DataFrame) -> float:
        """Local proxy: dispersion of comment↔post cosine similarity ∈ [0,1].

        Tight agreement around the post's framing ⇒ low spread ⇒ low polarization.
        """
        sims = metrics["context_cosine_similarity"].to_numpy(dtype=float)
        if sims.size < 2:
            return 0.5
        # std of cosine sims is bounded; scale to [0,1] with a soft cap at 0.5 std.
        return float(np.clip(np.std(sims) / 0.5, 0.0, 1.0))

    def run(
        self,
        enriched_comments: pd.DataFrame,
        post_context: pd.DataFrame,
        raw_comments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate to one row per ``media_id`` with community-vibe features."""
        LOGGER.info("Phase 3: aggregating community vibe over %d media.",
                    post_context[CANON_MEDIA_ID].nunique())

        # --- deterministic structural aggregates (vectorized group-by) -------
        agg = enriched_comments.groupby(CANON_MEDIA_ID).agg(
            comment_count=(CANON_COMMENT_ID, "count"),
            mean_cosine_similarity=("context_cosine_similarity", "mean"),
            mean_entity_overlap=("entity_overlap_count", "mean"),
            mean_emoji_density=("emoji_density", "mean"),
            mean_punctuation_velocity=("punctuation_velocity", "mean"),
            mean_noun_jaccard=("noun_jaccard_vs_post", "mean"),
        )

        # Aggregate audience noun phrases per media for structural Jaccard.
        comment_nouns: Dict[str, set[str]] = {}
        if "_noun_set" in enriched_comments.columns:
            for mid, sub in enriched_comments.groupby(CANON_MEDIA_ID)["_noun_set"]:
                bag: set[str] = set()
                for s in sub:
                    bag |= set(s)
                comment_nouns[str(mid)] = bag

        ctx_indexed = post_context.set_index(CANON_MEDIA_ID)
        raw_by_media = dict(tuple(raw_comments.groupby(CANON_MEDIA_ID)))

        vibe_records: List[Dict[str, Any]] = []
        for media_id, struct in agg.iterrows():
            mid = str(media_id)
            context = ctx_indexed.loc[media_id] if media_id in ctx_indexed.index else pd.Series(dtype=object)
            metrics_subset = enriched_comments[enriched_comments[CANON_MEDIA_ID] == mid]

            # Structural topical adherence: audience nouns ∩ post nouns (Jaccard).
            post_nouns = self.nlp.noun_set(str(context.get(CANON_POST_TEXT, "") or ""))
            topical_adherence = self.nlp.jaccard(comment_nouns.get(mid, set()), post_nouns)

            # --- probabilistic vibe (Gemini 2.5 Pro) with local fallback -----
            polarization = self._polarization_fallback(metrics_subset)
            dominant_emotion = "mixed"
            community_phrases: List[str] = sorted(comment_nouns.get(mid, set()))[:15]

            group = raw_by_media.get(media_id)
            if (
                group is not None
                and len(group) >= self.config.min_comments_for_vibe
            ):
                sample = self._sample_comments(group, self.config.max_comments_sampled)
                payload = self.llm.assess_community(self._build_prompt(sample, context))
                if payload is not None:
                    vibe = self._validate(payload)
                    polarization = vibe["sentiment_polarization_index"]
                    dominant_emotion = vibe["dominant_community_emotion"]
                    if vibe["community_noun_phrases"]:
                        # Refine adherence with LLM-surfaced phrases.
                        llm_nouns = {p.lower() for p in vibe["community_noun_phrases"]}
                        topical_adherence = max(
                            topical_adherence, self.nlp.jaccard(llm_nouns, post_nouns)
                        )
                        community_phrases = vibe["community_noun_phrases"]

            record = {
                CANON_MEDIA_ID: mid,
                CANON_PLATFORM: context.get(CANON_PLATFORM, pd.NA),
                "sentiment_polarization_index": polarization,
                "topical_adherence_score": topical_adherence,
                "dominant_community_emotion": dominant_emotion,
                "community_noun_phrases": community_phrases,
            }
            record.update(struct.to_dict())
            vibe_records.append(record)

        vibe_df = pd.DataFrame.from_records(vibe_records)
        LOGGER.info("Phase 3: built community vibe for %d media.", len(vibe_df))
        return vibe_df

    def _validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if _HAS_PYDANTIC:
            try:
                return CommunityVibe(**payload).model_dump()
            except ValidationError as exc:
                LOGGER.warning("CommunityVibe validation failed (%s); coercing.", exc)
        return {
            "sentiment_polarization_index": float(payload.get("sentiment_polarization_index", 0.5)),
            "dominant_community_emotion": str(payload.get("dominant_community_emotion", "mixed")),
            "community_noun_phrases": list(payload.get("community_noun_phrases", []) or []),
        }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class EnrichmentPipeline:
    """End-to-end driver: normalize → Phase 1 → Phase 2 → Phase 3 → parquet."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.llm = VertexLLMClient(self.config)
        self.nlp = LocalNLPEnricher(self.config)
        self.summarizer = PostContextSummarizer(self.config, self.llm)
        self.aggregator = CommunityVibeAggregator(self.config, self.llm, self.nlp)

    @staticmethod
    def _concat_normalized(
        frames: Dict[str, pd.DataFrame], normalizer
    ) -> pd.DataFrame:
        parts = [normalizer(df, platform) for platform, df in frames.items() if df is not None and len(df)]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def run(
        self,
        media_frames: Dict[str, pd.DataFrame],
        comment_frames: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        LOGGER.info("=== Enrichment pipeline START ===")

        # --- Normalization ---------------------------------------------------
        media = self._concat_normalized(media_frames, SchemaNormalizer.normalize_media)
        comments = self._concat_normalized(comment_frames, SchemaNormalizer.normalize_comments)
        if media.empty:
            raise ValueError("No media rows after normalization — aborting.")
        LOGGER.info("Normalized: %d media, %d comments.", len(media), len(comments))

        # --- Phase 1 ---------------------------------------------------------
        post_context = self.summarizer.run(media)

        # --- Phase 2 ---------------------------------------------------------
        if comments.empty:
            LOGGER.warning("No comments available; Phase 2/3 will yield empty community signal.")
            enriched_comments = pd.DataFrame(
                columns=[CANON_MEDIA_ID, CANON_COMMENT_ID, "context_cosine_similarity",
                         "entity_overlap_count", "emoji_density", "punctuation_velocity",
                         "noun_jaccard_vs_post"]
            )
        else:
            enriched_comments = self.nlp.enrich_comments(comments, post_context)

        # --- Phase 3 ---------------------------------------------------------
        vibe = (
            self.aggregator.run(enriched_comments, post_context, comments)
            if not enriched_comments.empty
            else pd.DataFrame(columns=[CANON_MEDIA_ID])
        )

        # --- Fuse: post context ⋈ community vibe -----------------------------
        matrix = post_context.merge(
            vibe, on=[CANON_MEDIA_ID], how="left", suffixes=("", "_vibe")
        )
        # Drop transient/large intermediate text to keep the matrix lean.
        matrix = matrix.drop(columns=[CANON_TRANSCRIPT], errors="ignore")
        self._persist(matrix)
        LOGGER.info(
            "=== Pipeline DONE === rows=%d | LLM calls=%d | LLM errors=%d",
            len(matrix), self.llm.call_count, self.llm.error_count,
        )
        return matrix

    def _persist(self, matrix: pd.DataFrame) -> None:
        out = self.config.output_path
        try:
            # List/array columns must be serialized for parquet portability.
            df = matrix.copy()
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, tuple, set))).any():
                    df[col] = df[col].apply(
                        lambda x: json.dumps(list(x), ensure_ascii=False)
                        if isinstance(x, (list, tuple, set)) else x
                    )
            df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
            LOGGER.info("Wrote enriched matrix → %s", out.resolve())
        except Exception as exc:
            LOGGER.error("Parquet write failed (%s); falling back to CSV.", exc)
            matrix.to_csv(out.with_suffix(".csv"), index=False)


# --------------------------------------------------------------------------- #
# Mock execution block
# --------------------------------------------------------------------------- #
def _mock_frames() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Synthetic but schema-faithful cross-platform data for a dry run."""
    ig_media = pd.DataFrame(
        {
            "media_id": ["IG1", "IG2"],
            "caption": [
                "Nuovo video con @nike! Oggi vi mostro la mia routine mattutina 🏃‍♀️ #ad",
                "Riflessioni della domenica... a volte basta poco per sorridere ☀️",
            ],
            "transcript": [
                "Ciao ragazzi oggi proviamo le nuove scarpe Nike durante la corsa al parco.",
                "",
            ],
            "timestamp": ["2026-03-01T10:00:00Z", "2026-03-02T18:30:00Z"],
        }
    )
    fb_posts = pd.DataFrame(
        {
            "post_id": ["FB1"],
            "message": ["Grande notizia: apriamo un nuovo negozio a Milano! Vi aspettiamo 🎉"],
            "timestamp": ["2026-03-03T09:00:00Z"],
        }
    )
    tk_media = pd.DataFrame(
        {
            "video_id": ["TK1"],
            "caption": ["POV: quando il caffè finisce 😂 #comedy"],
            "timestamp": ["2026-03-04T12:00:00Z"],
        }
    )

    ig_comments = pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3", "c4"],
            "media_id": ["IG1", "IG1", "IG1", "IG2"],
            "from_id": ["u1", "u2", "u3", "u4"],
            "text": [
                "Adoro le Nike! Anche io corro ogni mattina ❤️🔥",
                "Bellissima routine, super motivante 💪",
                "Troppa pubblicità ultimamente... 🙄",
                "Che dolce ☀️ buona domenica!",
            ],
            "parent_id": [None, None, "c1", None],
            "timestamp": ["2026-03-01T11:00:00Z"] * 4,
        }
    )
    fb_comments = pd.DataFrame(
        {
            "comment_id": ["f1", "f2", "f3"],
            "post_id": ["FB1", "FB1", "FB1"],
            "from_id": ["u5", "u6", "u7"],
            "message": [
                "Finalmente a Milano! Verrò sicuramente 🎉",
                "Speriamo aprano anche a Roma",
                "Non mi interessa per niente",
            ],
            "parent_id": [None, None, None],
            "timestamp": ["2026-03-03T10:00:00Z"] * 3,
        }
    )
    tk_comments = pd.DataFrame(
        {
            "comment_id": ["t1", "t2", "t3"],
            "video_id": ["TK1", "TK1", "TK1"],
            "uid": ["u8", "u9", "u10"],
            "text": ["HAHAHA è successo anche a me 😂", "Troppo vero 💀", "Il caffè è vita ☕"],
            "reply_id": [None, "t1", None],
            "timestamp": ["2026-03-04T13:00:00Z"] * 3,
        }
    )

    media_frames = {"instagram": ig_media, "facebook": fb_posts, "tiktok": tk_media}
    comment_frames = {"instagram": ig_comments, "facebook": fb_comments, "tiktok": tk_comments}
    return media_frames, comment_frames


def main() -> None:
    LOGGER.info("Running Show Reel enrichment pipeline in MOCK mode.")
    media_frames, comment_frames = _mock_frames()

    # enable_llm=False forces deterministic local-only execution for the demo.
    # Flip to True in production once ADC / Vertex AI access is configured.
    config = PipelineConfig(
        enable_llm=False,
        output_path=Path("enriched_post_vibe_matrix.parquet"),
    )
    pipeline = EnrichmentPipeline(config)
    matrix = pipeline.run(media_frames, comment_frames)

    pd.set_option("display.max_columns", None, "display.width", 200)
    LOGGER.info("Resulting enriched_post_vibe_matrix:\n%s", matrix.to_string(index=False))


if __name__ == "__main__":
    main()
