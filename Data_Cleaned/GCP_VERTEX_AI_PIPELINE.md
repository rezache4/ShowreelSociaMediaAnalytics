# GCP Vertex AI LLM Ingestion Pipeline Design

**Document Version:** 1.0  
**Date:** May 17, 2026  
**Scope:** Ingest ~11,801 YouTube video transcripts into Google Vertex AI for LLM processing (summarization, extraction, QA, or custom analysis)  
**Status:** Design Phase (Ready for Implementation)

---

## 1. Executive Summary

This pipeline automates the ingestion, preprocessing, and batch processing of YouTube transcripts via Google Cloud Vertex AI. Key features:

- **Input:** Parquet file (`yt_videos_with_local_transcripts.parquet`) with 11,801 rows, `local_transcript` column
- **Processing:** Local cleaning, deduplication, token-aware chunking, manifest creation, and optional language detection before upload
- **Output:** Prompt-ready job bundles and idempotent inference results stored in BigQuery or GCS
- **Scale:** Handles 11K+ transcripts with batching, retries, rate-limiting, token budgeting, and job manifests
- **Timeline:** ~2–5 hours for full run, with most deterministic work done locally

## 1.1 Final Architecture Decision

After reviewing the Italian NLP research note, the best fit for this project is a **lightweight transcript-first pipeline on GCP**, not a heavy tokenizer-adaptation workflow.

Why this is the right tradeoff:

- The research note is optimized for **noisy, informal social-media comments** across platforms, where clitic splitting, hashtag segmentation, profanity restoration, and emoji semantic parsing materially improve quality.
- Our current workload is **YouTube transcripts** that are already much cleaner than raw comments, so most of that preprocessing would add cost and complexity without enough return.
- We do **not** have the local hardware to train or adapt large open-weight models efficiently, so the workflow should minimize local compute and push expensive steps to managed GCP services.

Final decision:

- Use **minimal normalization** locally: lowercase, whitespace cleanup, timestamp cleanup, transcript deduplication, and optional language detection.
- Use **token-aware local chunking** so the cloud only sees compact, prompt-ready units.
- Use **Vertex AI managed embedding models** first, specifically `text-multilingual-embedding-002` or `gemini-embedding-001`, because they are serverless and remove GPU ops burden.
- Use **Gemini Flash-class LLMs** for downstream transcript analysis tasks that need generation, extraction, or classification.
- Avoid custom tokenizer surgery, SAVA/FVT adaptation, or self-hosted embedding stacks unless a later evaluation shows a measurable quality gap that justifies the extra infrastructure cost.

In short: **opt for serverless GCP models plus light preprocessing now; reserve open-weight or tokenizer adaptation work for a later optimization phase only if needed.**

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Local Data (Workspace)                                          │
│  ├─ yt_videos_with_local_transcripts.parquet (input)           │
│  ├─ prepare_transcript_jobs.py (local preprocessor)            │
│  └─ config.json (credentials, settings)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
       ┌──────────────────────────────────────┐
       │ Step 1: Local job assembly           │
       │  • Clean transcript text             │
       │  • Deduplicate by content hash       │
       │  • Token-aware chunking              │
       │  • Manifest + skip/reject records     │
       └──────────────┬───────────────────────┘
                 │
                 ▼
       ┌──────────────────────────────┐
       │ Step 2: Upload compact JSONL │
       │ to GCS (job bundles only)    │
       └──────────────┬───────────────┘
                 │
                 ▼
   ┌──────────────────────────────────────────────┐
   │ Step 3: Cloud Run Job / Vertex Batch Worker   │
   │  • Read job bundle from GCS                   │
   │  • Call Vertex AI only for inference          │
   │  • Write idempotent outputs keyed by job_id   │
   │  • Retry safely using manifest checkpoints    │
   └──────────────┬───────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────┐
  │ Step 4: Results Storage & Monitoring│
  │ • BigQuery (final results)          │
  │ • GCS (raw responses + manifests)   │
  │ • Cloud Logging (metrics, errors)   │
  └─────────────────────────────────────┘
```

### 2.1 Local vs GCP Responsibility Split

The default path keeps all deterministic, cheap, and non-sequential work local. Only inference and durable storage go to GCP.

| Layer | Run Locally | Send to GCP |
| :---- | :---- | :---- |
| Transcript cleanup | Timestamp/tag stripping, whitespace cleanup, normalization | No |
| Deduplication | Content-hash dedupe, empty-row rejection | No |
| Token budgeting | Token estimation and chunk sizing | No |
| Chunk assembly | Build prompt-ready JSONL records | No |
| Language detection | Optional metadata field | No |
| Inference | No | Yes, only the final compact prompt payload |
| Result persistence | Local manifest and staging files | BigQuery/GCS |
| Aggregation | Lightweight final merge after cloud return | Optional |

This split reduces cloud token burn because the model only receives cleaned, deduplicated, size-controlled inputs.

---

## 3. Data Flow & Stages

### 3.1 Stage 1: Prepare & Upload Data

**Input:** `yt_videos_with_local_transcripts.parquet`  
**Output:** `gs://showreel-bucket/data/transcripts_raw.jsonl`

Script: `upload_to_gcs.py`

```python
# Pseudo-code
def prepare_and_upload():
    df = pd.read_parquet('yt_videos_with_local_transcripts.parquet')
    
    # Select relevant columns
    df_subset = df[['videoId', 'title', 'local_transcript']].copy()
    
    # Filter out rows without transcripts
    df_subset = df_subset[df_subset['local_transcript'].notna()]
    
    # Write to JSONL (one object per line)
    df_subset.to_json('transcripts_raw.jsonl', orient='records', lines=True)
    
    # Upload to GCS
    bucket = storage.Client().bucket('showreel-bucket')
    blob = bucket.blob('data/transcripts_raw.jsonl')
    blob.upload_from_filename('transcripts_raw.jsonl')
    
    return df_subset.shape[0]  # Report row count
```

**Considerations:**
- Use streaming upload for large files
- Validate JSONL structure before upload
- Log upload timestamp and hash

---

### 3.2 Stage 2: Preprocess & Chunk Locally

**Input:** `yt_videos_with_local_transcripts.parquet`  
**Output:** `Data_Cleaned/gcp_jobs/transcript_jobs.jsonl` and `Data_Cleaned/gcp_jobs/transcript_jobs_manifest.json`

Script: `prepare_transcript_jobs.py` (local)

Recommended preprocessing scope for this project:

- Normalize transcript text with the existing `strip_timestamps_and_tags()` style cleanup.
- Collapse repeated whitespace and remove obvious markup artifacts.
- Keep language detection as a metadata field, not a hard filter.
- Skip heavy Italian-specific operations from the research note because the current input is transcripts, not raw comments.
- Do not invest in tokenizer re-training or embedding-space mapping for the first production pass.
- Build prompt-ready records with `job_id`, `videoId`, `chunk_idx`, `chunk_total`, `estimated_input_tokens`, and `source_hash`.

That keeps preprocessing cheap while preserving the transcript semantics that matter for downstream analysis, while also giving us retry-safe job boundaries.

**Output JSONL Structure:**
```json
{"job_id": "eNU5FcmEwTc_000", "video_id": "eNU5FcmEwTc", "chunk_idx": 0, "chunk_total": 1, "chunk_text": "...", "estimated_input_tokens": 1856, "language": "it", "source_hash": "...", "prompt_version": "v1"}
{"job_id": "eNU5FcmEwTc_001", "video_id": "eNU5FcmEwTc", "chunk_idx": 1, "chunk_total": 3, "chunk_text": "...", "estimated_input_tokens": 2000, "language": "it", "source_hash": "...", "prompt_version": "v1"}
```

**Deployment Options:**
- **Default:** local script + GCS upload + Cloud Run Job / Vertex batch worker
- **Fallback:** Cloud Function only if a future source becomes event-driven and tiny
- **Not recommended for the default path:** Pub/Sub fan-out, because it adds operational overhead without improving the current batch workload

---

### 3.3 Stage 3: Upload Job Bundles to GCS

**Input:** `Data_Cleaned/gcp_jobs/transcript_jobs.jsonl`  
**Output:** `gs://showreel-bucket/data/transcript_jobs.jsonl`

Script: `upload_jobs_to_gcs.py`

```python
from google.cloud import storage

def upload_job_bundle(local_jsonl_path, bucket_name, object_name):
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(object_name)
  blob.upload_from_filename(local_jsonl_path)
  return f"gs://{bucket_name}/{object_name}"
```

**Why this is better than Pub/Sub here:**
- The workload is already batch-oriented.
- The local stage can assemble, validate, and compress the jobs before upload.
- A single GCS object plus manifest gives stronger replayability than a stream of transient messages.

---

### 3.4 Stage 4: Process with Vertex AI LLM (Cloud Run Job or Vertex Batch)

**Input:** GCS job bundle (`transcript_jobs.jsonl`)  
**Output:** BigQuery table `showreel_dataset.inference_results`

Script: `worker.py` (Cloud Run Job)

```python
from google.cloud import bigquery, aiplatform
from google.api_core import retry
import json
import os
import time

PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'showreel-proj')
REGION = 'us-central1'
BQ_DATASET = 'showreel_dataset'
BQ_TABLE = 'inference_results'

# Initialize clients
bq_client = bigquery.Client(project=PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=REGION)

def call_vertex_ai_llm(chunk_text, video_id, chunk_idx):
    """Call Vertex AI LLM with a transcript chunk (example: summarization)."""
  model = GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""Summarize the following transcript in 2-3 sentences, highlighting key topics:

Transcript:
{chunk_text}

Summary:"""
    
    response = model.generate_content(
      prompt,
      generation_config={
        "max_output_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
      },
    )
    
    return {
        'video_id': video_id,
        'chunk_idx': chunk_idx,
        'summary': response.text,
        'model': 'gemini-1.5-flash',
        'tokens_input': len(chunk_text.split()),
        'timestamp': int(time.time())
    }

@retry.Retry(deadline=300)
def insert_to_bigquery(rows):
    """Insert inference results to BigQuery with idempotent keys."""
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    table = bq_client.get_table(table_id)
    errors = bq_client.insert_rows_json(table, rows)
    
    if errors:
        raise Exception(f"BigQuery insert errors: {errors}")

def process_job_record(job_record):
  """Process a single job record from the local bundle."""
    try:
    video_id = job_record['video_id']
    chunk_idx = job_record['chunk_idx']
    chunk_text = job_record['chunk_text']
    job_id = job_record['job_id']

        # Call Vertex AI LLM
        result = call_vertex_ai_llm(chunk_text, video_id, chunk_idx)
    result['job_id'] = job_id
    result['prompt_version'] = job_record.get('prompt_version', 'v1')

        # Insert to BigQuery
        insert_to_bigquery([result])

        print(f"✓ Processed {video_id} chunk {chunk_idx}")
        return True
    
    except Exception as e:
    print(f"✗ Error processing job record: {e}")
        return False

if __name__ == '__main__':
  print('Run as a Cloud Run Job or batch worker that reads a GCS JSONL bundle.')
```

**Cloud Run Deployment:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY worker.py .

CMD ["python", "worker.py"]
```

**Reliability rules:**
- Every job record must have a stable `job_id` and `source_hash`.
- Every output row must be keyed by `job_id` + `prompt_version`.
- Jobs should be safe to rerun without creating duplicate semantic records.
- Failures should be written to a retry manifest instead of silently dropped.

---

### 3.5 Stage 5: Results Storage & Monitoring

**BigQuery Schema:**

```sql
CREATE TABLE `showreel-proj.showreel_dataset.inference_results` (
  video_id STRING NOT NULL,
  chunk_idx INT64 NOT NULL,
  summary STRING,
  model STRING,
  tokens_input INT64,
  timestamp INT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  cost_usd FLOAT64,
  latency_ms INT64
);

CREATE TABLE `showreel-proj.showreel_dataset.processing_metrics` (
  run_id STRING,
  total_chunks INT64,
  processed_chunks INT64,
  failed_chunks INT64,
  total_tokens_input INT64,
  total_tokens_output INT64,
  total_cost_usd FLOAT64,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  duration_seconds INT64
);
```

**Monitoring & Logging:**

```python
import google.cloud.logging

def setup_logging():
    """Configure Cloud Logging."""
    log_client = google.cloud.logging.Client()
    log_client.setup_logging()

# Then use standard Python logging:
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing chunk {chunk_idx} from video {video_id}")
logger.warning(f"High latency: {latency_ms}ms")
logger.error(f"API error: {error_msg}")
```

**Cloud Monitoring Alerts:**
- Batch backlog > 1000 pending jobs → scale workers
- Error rate > 5% → investigate failures
- Cost per hour > threshold → pause processing

---

## 4. Security & IAM

### 4.1 Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create showreel-pipeline \
  --display-name="Showreel Transcript Pipeline"

# Grant necessary roles
gcloud projects add-iam-policy-binding showreel-proj \
  --member=serviceAccount:showreel-pipeline@showreel-proj.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
  # (read/write GCS)

gcloud projects add-iam-policy-binding showreel-proj \
  --member=serviceAccount:showreel-pipeline@showreel-proj.iam.gserviceaccount.com \
  --role=roles/bigquery.dataEditor
  # (write to BigQuery)

gcloud projects add-iam-policy-binding showreel-proj \
  --member=serviceAccount:showreel-pipeline@showreel-proj.iam.gserviceaccount.com \
  --role=roles/aiplatform.user
  # (call Vertex AI APIs)
```

### 4.2 Data Encryption

- **GCS:** Enable default encryption (Google-managed keys) or CMEK (Customer-managed encryption keys)
- **BigQuery:** Encrypted at rest by default; enable audit logging
- **Cloud Run:** Use service account with minimal permissions; enable workload identity

### 4.3 Access Control

- Restrict access to `showreel-pipeline` service account only
- Enable Cloud Audit Logs for all data access
- Use VPC Service Controls to enforce data residency and prevent exfiltration

---

## 5. Cost Estimation

### 5.1 Per-Run Costs (11,801 videos)

| Component | Estimate | Unit | Cost |
|-----------|----------|------|------|
| Vertex AI LLM (Gemini Flash-class) | 80M tokens | $0.25/1M input | $20 |
| GCS (upload, download) | 5 GB | $0.020/GB | $0.10 |
| Cloud Run Job / Batch Worker | 5,000 vCPU-h | $0.00002400/vCPU-s | $432 |
| BigQuery (storage) | 10 GB | $0.025/GB/mo | $0.25 |
| Cloud Logging | 500 MB | $0.50/GB | $0.25 |
| **Total per run** | | | **$452** |

### 5.2 Cost Optimization

1. **Use batch processing:** Call Vertex AI with 10–50 chunks per request
2. **Reduce token count:** Summarize or truncate transcripts before sending
3. **Implement caching:** Reuse outputs for identical transcripts and prompt versions
4. **Set budget alerts:** GCP billing alerts at $200/day

---

## 6. Prototype Implementation Plan

### Phase 1: Minimal Viable Pipeline (Day 1)

**Goal:** End-to-end test with 100 transcripts

1. Create GCP project, enable APIs, set up service account
2. Run the local job builder on 100 sample transcripts
3. Validate the local manifest, token estimates, and chunk counts
4. Upload one compact JSONL bundle to GCS
5. Process it with one Cloud Run Job or batch worker
6. Insert 1 sample result to validate schema

**Scripts:**
- `01_setup_gcp.sh` (create bucket, BigQuery dataset, service account)
- `02_prepare_transcript_jobs.py` (local clean + dedupe + chunk + manifest)
- `03_upload_jobs_to_gcs.py` (upload compact job bundle)
- `04_test_llm_call.py` (worker smoke test on one chunk)

### Phase 2: Automated Queue Processing (Day 1–2)

**Goal:** Process 1000 transcripts via batch job with retry-safe manifests

1. Split the local bundle into batch-sized shards
2. Deploy a Cloud Run Job or batch worker
3. Process each shard, log results to BigQuery
4. Retry only failed shards using the manifest
5. Monitor for errors, retries, cost

**Scripts:**
- `05_split_batches.py`
- `06_cloud_run_worker/` (Docker + batch runner)
- `07_monitor_progress.py` (query BigQuery for stats)

### Phase 3: Full-Scale Production (Day 2–3)

**Goal:** Process all 11,801 transcripts

1. Scale Cloud Run replicas to 5–10
2. Implement exponential backoff + max retry logic
3. Add cost tracking per video
4. Generate final report (summary counts, errors, total cost)

**Scripts:**
- `08_scale_deployment.sh` (gcloud run deploy with --concurrency, --max-instances)
- `09_cost_report.py` (BigQuery analysis)

### Phase 4: Post-Processing & Cleanup (Day 3)

**Goal:** Join results back to original parquet

1. Export BigQuery results to Parquet
2. Merge with `yt_videos_with_local_transcripts.parquet` on `video_id`
3. Clean up GCS staging files only after row-count reconciliation
4. Archive pipeline logs and retry manifests

**Scripts:**
- `10_export_results.py` (BigQuery → Parquet)
- `11_merge_results.py` (join with original data)
- `12_cleanup.py` (delete staging files)

---

## 7. Reliability and Token-Budget Rules

These rules are there to avoid later rework:

1. **Only send prompt-ready text to GCP.** Clean, deduplicated, token-budgeted records are built locally.
2. **Use stable identifiers.** Every transcript chunk needs `job_id`, `source_hash`, `prompt_version`, and `videoId`.
3. **Keep outputs idempotent.** Reprocessing the same job should overwrite or skip by key instead of duplicating rows.
4. **Keep prompts short and fixed.** Do not rebuild prompt text inside the cloud worker for every job if the instruction is constant.
5. **Prefer batch execution over stream fan-out.** Batch jobs are easier to resume and reconcile.
6. **Record rejection reasons locally.** Empty, duplicate, or too-short transcripts should be written to a local reject file.
7. **Validate counts before upload.** Local manifest counts should match the number of GCS job records and the final BigQuery inserts.

---

## 8. Configuration File (`config.json`)

```json
{
  "gcp": {
    "project_id": "showreel-proj",
    "region": "us-central1",
    "service_account_email": "showreel-pipeline@showreel-proj.iam.gserviceaccount.com"
  },
  "gcs": {
    "bucket_name": "showreel-bucket",
    "input_prefix": "data/",
    "job_prefix": "jobs/",
    "output_prefix": "results/"
  },
  "bigquery": {
    "dataset_id": "showreel_dataset",
    "table_id": "inference_results"
  },
  "processing": {
    "chunk_size_tokens": 1800,
    "chunk_overlap_tokens": 150,
    "batch_size": 10,
    "max_workers": 10,
    "prompt_version": "v1"
  },
  "vertex_ai": {
    "embedding_model": "gemini-embedding-001",
    "llm_model": "gemini-1.5-flash",
    "temperature": 0.2,
    "max_output_tokens": 256,
    "max_input_tokens": 1800
  },
  "cost_budget": {
    "max_cost_per_hour_usd": 200.0,
    "alert_threshold_usd": 50.0
  }
}
```

---

## 9. Next Steps

1. **Review & Approve:** Validate design with stakeholders
2. **Create GCP Resources:** Run `01_setup_gcp.sh` to initialize cloud infrastructure
3. **Implement Phase 1:** Test with 100 transcripts (1 day)
4. **Iterate & Scale:** Move to Phase 2–3 based on Phase 1 results
5. **Monitor & Report:** Use BigQuery dashboards for real-time progress

---

## 10. Appendix: Sample Prompt Ideas

Depending on use case, customize the LLM prompt:

### Option A: Summarization
```
Summarize the following transcript in 2–3 sentences:
{chunk_text}
```

### Option B: Entity Extraction
```
Extract the main topics, people, and organizations mentioned:
{chunk_text}
Format as JSON: {"topics": [...], "people": [...], "organizations": [...]}
```

### Option C: Sentiment Analysis
```
Classify the sentiment of this transcript (positive/neutral/negative) and explain:
{chunk_text}
```

### Option D: Custom Q&A
```
Given the transcript, answer: "What are the main technical concepts discussed?"
{chunk_text}
```

---

## 11. References & Resources

- [Vertex AI LLM API Docs](https://cloud.google.com/vertex-ai/docs/generative-ai)
- [Cloud Run Deployment Guide](https://cloud.google.com/run/docs/deploying)
- [BigQuery Schema Design](https://cloud.google.com/bigquery/docs/schemas)
- [Cost Optimization Best Practices](https://cloud.google.com/architecture/cost-optimization-checklist)
