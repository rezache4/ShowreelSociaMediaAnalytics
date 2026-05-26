# Antigravity Quickstart

This repo is a transcript/YouTube analysis workspace. The fastest way to make an agent productive is to point it at these files first:

- `.agent.md` for the repo-specific agent role and scope
- `Data_Cleaned/GCP_VERTEX_AI_PIPELINE.md` for the current local-first GCP plan
- `Data_Cleaned/pipeline_tools/add_transcripts.py` for transcript merge logic
- `Data_Cleaned/pipeline_tools/prepare_transcript_jobs.py` for local job bundle creation
- `Data_Cleaned/Youtube_channels_README.md` for the metadata schema

## What to tell the agent

Use this as the first instruction block:

> You are working in a Windows workspace for transcript processing and GCP batch inference.
> Focus on `Data_Cleaned/` first.
> Do not deduplicate transcripts unless explicitly asked.
> Keep preprocessing lightweight: timestamp stripping, whitespace cleanup, optional language metadata.
> Use local preprocessing and push only compact job bundles to GCP.
> Read `.agent.md` and `Data_Cleaned/GCP_VERTEX_AI_PIPELINE.md` before editing.

## Fast setup

1. Open the workspace root in Antigravity IDE.
2. Open `.agent.md` and `Data_Cleaned/GCP_VERTEX_AI_PIPELINE.md`.
3. Point the agent to `Data_Cleaned/pipeline_tools/`.
4. Make sure the Python interpreter is the Conda env at `D:/miniconda/envs/showreel-kernel/python.exe`.
5. If you need the transcript parquet, use `Data_Cleaned/yt_videos_with_local_transcripts.parquet`.

## Useful commands

```bash
python Data_Cleaned/pipeline_tools/add_transcripts.py --input Data_Cleaned/yt_videos_cleaned.parquet --output Data_Cleaned/yt_videos_with_local_transcripts.parquet
python Data_Cleaned/pipeline_tools/prepare_transcript_jobs.py --input Data_Cleaned/yt_videos_with_local_transcripts.parquet --output Data_Cleaned/gcp_jobs/transcript_jobs.jsonl --manifest Data_Cleaned/gcp_jobs/transcript_jobs_manifest.json
```

Use `--dedupe` only if you really want content-hash deduplication.

## Current project shape

- Input data lives in `Data_Cleaned/`
- Recycled/removed files are already logged in `Data_Cleaned/recycle_manifest.json` and `Data_Cleaned/recycle_manifest_code_cleanup.json`
- The current pipeline is local-first, then GCP batch inference
- The best cloud models documented here are Vertex AI Gemini-class models and embedding models

## If the agent needs to inspect quickly

Start with these three files in order:

1. `.agent.md`
2. `Data_Cleaned/GCP_VERTEX_AI_PIPELINE.md`
3. `Data_Cleaned/pipeline_tools/prepare_transcript_jobs.py`

That is enough to understand the project within minutes.
