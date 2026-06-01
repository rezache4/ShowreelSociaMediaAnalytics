"""Rebuild community_vibe_pipeline_colab.ipynb from the CURRENT source files.

Bundles prompts.py as a `%%writefile` cell so the module cell's
`from prompts import ...` resolves on Colab Enterprise.
"""
import json
from pathlib import Path

SRC = Path("community_vibe_pipeline.py").read_text(encoding="utf-8")
PROMPTS_SRC = Path("prompts.py").read_text(encoding="utf-8")

# --- Adaptation 1: output path default -> GCS URI (str, not Path) ----------
SRC = SRC.replace(
    '    output_path: Path = Path("enriched_post_vibe_matrix.parquet")',
    '    output_path: str = "gs://afb_showreel/enriched/enriched_post_vibe_matrix.parquet"',
)

# --- Adaptation 2: GCS-aware _persist (pandas+gcsfs writes gs:// directly) --
OLD_PERSIST = '''    def _persist(self, matrix: pd.DataFrame) -> None:
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
            matrix.to_csv(out.with_suffix(".csv"), index=False)'''

NEW_PERSIST = '''    def _persist(self, matrix: pd.DataFrame) -> None:
        # Colab disk is ephemeral: write straight to GCS via pandas + gcsfs.
        out = str(self.config.output_path)
        df = matrix.copy()
        # List/array columns must be serialized for parquet portability.
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, tuple, set))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(list(x), ensure_ascii=False)
                    if isinstance(x, (list, tuple, set)) else x
                )
        try:
            df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
            LOGGER.info("Wrote enriched matrix → %s", out)
        except Exception as exc:
            csv_out = (out[:-8] + ".csv") if out.endswith(".parquet") else (out + ".csv")
            LOGGER.error("Parquet write failed (%s); falling back to CSV → %s", exc, csv_out)
            df.to_csv(csv_out, index=False)'''

assert OLD_PERSIST in SRC, "persist block not found verbatim"
SRC = SRC.replace(OLD_PERSIST, NEW_PERSIST)

# --- Adaptation 3: keep _mock_frames, drop interactive main()/__main__ ------
cut = SRC.index("\ndef main() -> None:")
MODULE_SRC = SRC[:cut].rstrip() + "\n"

# --------------------------------------------------------------------------- #
# Cell sources
# --------------------------------------------------------------------------- #
MD_TITLE = """# Show Reel — Post-Level Context & Community Vibe (Colab Enterprise)

Headless, scheduled-executor build of `community_vibe_pipeline.py`.

**Steps:** Post-Level Context Enrichment (Gemini 2.5 Flash) → Local NLP
Enrichment (spaCy `it_core_news_lg`) → Community Vibe & Polarization
(Gemini 2.5 Pro) → `enriched_post_vibe_matrix.parquet` on GCS.

Runs top-to-bottom, no interactivity. Auth = Application Default Credentials
(the Colab Enterprise service account). Reads raw CSVs from
`gs://afb_showreel/raw/`, writes the fused matrix to `gs://afb_showreel/enriched/`.

All LLM prompts live in the `prompts.py` cell below (single source of truth)."""

CELL_PIP = """# Cell 1 — dependencies (non-builtin only)
!pip install -q google-cloud-aiplatform gcsfs pyarrow spacy emoji pydantic"""

CELL_SPACY = """# Cell 2 — Italian spaCy model (has word vectors -> needed for cosine similarity)
!python -m spacy download it_core_news_lg"""

CELL_CONFIG = '''# Cell 3 — config + logging (hoisted; authoritative for the whole notebook)
import logging
import sys

# --- GCP config (Colab Enterprise / Vertex AI) -----------------------------
GCP_PROJECT_ID = "gen-lang-client-0792749758"
GCP_BUCKET = "afb_showreel"
GCP_LOCATION = "us-central1"

# --- GCS layout ------------------------------------------------------------
RAW_PREFIX = f"gs://{GCP_BUCKET}/raw"                       # raw input CSVs
OUTPUT_URI = f"gs://{GCP_BUCKET}/enriched/enriched_post_vibe_matrix.parquet"

# --- Structured, stdout logging the scheduler can capture ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("community_vibe_pipeline").info(
    "Config loaded: project=%s bucket=%s location=%s", GCP_PROJECT_ID, GCP_BUCKET, GCP_LOCATION
)'''

# %%writefile MUST be the first line of the cell -> materialize prompts.py so
# the module cell's `from prompts import ...` resolves on Colab.
CELL_PROMPTS = "%%writefile prompts.py\n" + PROMPTS_SRC

CELL_MODULE = MODULE_SRC  # the faithful pipeline (classes + _mock_frames)

CELL_LOADER = '''# Cell 6 — GCS data loader (pandas reads gs:// natively via gcsfs)
# Expected raw files under RAW_PREFIX. Adjust filenames to match your bucket.
MEDIA_FILES = {"instagram": "ig_media.csv", "facebook": "fb_posts.csv", "tiktok": "tk_media.csv"}
COMMENT_FILES = {"instagram": "ig_comments.csv", "facebook": "fb_comments.csv", "tiktok": "tk_comments.csv"}


def load_frames_from_gcs():
    """Load per-platform media + comment CSVs from GCS; skip-and-log missing."""
    media, comments = {}, {}
    for plat, fn in MEDIA_FILES.items():
        uri = f"{RAW_PREFIX}/{fn}"
        try:
            media[plat] = pd.read_csv(uri)
            LOGGER.info("[%s] media: %d rows from %s", plat, len(media[plat]), uri)
        except Exception as exc:
            LOGGER.warning("[%s] media unavailable (%s): %s", plat, uri, exc)
    for plat, fn in COMMENT_FILES.items():
        uri = f"{RAW_PREFIX}/{fn}"
        try:
            comments[plat] = pd.read_csv(uri)
            LOGGER.info("[%s] comments: %d rows from %s", plat, len(comments[plat]), uri)
        except Exception as exc:
            LOGGER.warning("[%s] comments unavailable (%s): %s", plat, uri, exc)
    return media, comments'''

CELL_RUN = '''# Cell 7 — run (fatal-error guard: log traceback then re-raise so the
# scheduled run is marked FAILED rather than silently "succeeding")
import traceback

try:
    media_frames, comment_frames = load_frames_from_gcs()

    if not any(len(d) for d in media_frames.values()):
        LOGGER.warning(
            "No raw media found under %s — using bundled MOCK frames for a smoke run.",
            RAW_PREFIX,
        )
        media_frames, comment_frames = _mock_frames()

    config = PipelineConfig(
        gcp_project_id=GCP_PROJECT_ID,
        gcp_location=GCP_LOCATION,
        output_path=OUTPUT_URI,
        enable_llm=True,   # Vertex AI is reachable on Colab Enterprise via ADC
    )
    pipeline = EnrichmentPipeline(config)
    matrix = pipeline.run(media_frames, comment_frames)
    LOGGER.info(
        "Pipeline finished OK — %d media rows | LLM calls=%d errors=%d → %s",
        len(matrix), pipeline.llm.call_count, pipeline.llm.error_count, OUTPUT_URI,
    )
except Exception:
    LOGGER.error("FATAL — pipeline aborted:\\n%s", traceback.format_exc())
    raise'''


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md_cell(MD_TITLE),
        code_cell(CELL_PIP),
        code_cell(CELL_SPACY),
        code_cell(CELL_CONFIG),
        md_cell("### Prompt definitions — written to `prompts.py` (edit prompts here)"),
        code_cell(CELL_PROMPTS),
        code_cell(CELL_MODULE),
        code_cell(CELL_LOADER),
        code_cell(CELL_RUN),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("community_vibe_pipeline_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} with {len(nb['cells'])} cells")
