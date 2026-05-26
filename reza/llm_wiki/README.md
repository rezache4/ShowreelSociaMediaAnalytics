# LLM Ingestion Pipeline Wiki

Welcome to the Showreel LLM Wiki. This directory documents the design, data structures, and reasoning behind our transcript preprocessing and GCP batch ingestion pipeline.

## Wiki Contents

- [Deterministic Preprocessing](local_deterministic_compute.md): Why we clean and chunk transcripts locally and what steps are executed.
- [JSONL & Cloud Batch Processing](jsonl_batch_format.md): Why JSONL is chosen, how it aligns with Vertex AI Batch APIs, and how chunk metadata is preserved.
- [Hashing & Content Tracking](hashing_and_deduplication.md): The role of SHA-256 content hashes, idempotency, and audit checks.

## Pipeline Architecture Overview

```
[Local Parquet Data] 
       │
       ▼ (Deterministic Compute)
[Text Clean -> Token Chunk -> SHA256 Hash]
       │
       ▼ (Local Assembly)
[JSONL Job Bundle + Manifest]
       │
       ▼ (GCS Upload)
[GCP Vertex AI Batch Inference]
```
