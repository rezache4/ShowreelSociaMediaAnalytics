# Git Push Vetting Plan
# ShowreelSocialMediaAnalytics → github.com/rezache4/ShowreelSociaMediaAnalytics

## Rule: code and docs go to git; data stays on OneDrive

---

## INCLUDED in git

### Root level
| File | Reason |
|------|--------|
| `ANTIGRAVITY_QUICKSTART.md` | Pipeline onboarding entry point |
| `italian_nlp_tools.md` | Architectural decisions for Italian NLP |
| `shorts_deduplication_test.ipynb` | Validated deduplication algorithm with real test results |
| `.agent.md` | Claude Code agent scope definition for transcript pipeline |
| `PUSH_PLAN.md` | This file |

### `llm_wiki/`
All 4 markdown files — architectural rationale for JSONL format, hashing, local compute, and pipeline overview.

### `Data_Cleaned/` — scripts and docs only
| File | Reason |
|------|--------|
| `GCP_VERTEX_AI_PIPELINE.md` | Full pipeline design and architecture decisions |
| `Youtube_channels_README.md` | Schema docs for the 27-channel metadata CSV |
| `youtube_category_mapping.json` | Small reference mapping (< 1 KB) |
| `Untitled-1.ipynb` | Active EDA notebook (YT video time series, channel comparisons) |
| `add_transcripts.py` | CLI entrypoint wrapper |
| `foreman.py` | Work distribution for parallel transcript download |
| `worker_script.py` | yt_dlp wrapper with retry logic |
| `trigger_queue.py` | Batch job queue utility |
| `generate_retry_queue.py` | Retry queue builder for failed transcript downloads |
| `match_cloud_transcripts.py` | Cloud vs local transcript reconciliation |
| `pool_and_final_check_transcripts.py` | Final validation pass on pooled transcripts |
| `prepare_transcript_jobs.py` | Root-level stub (delegates to pipeline_tools/) |
| `pipeline_tools/add_transcripts.py` | Core transcript merge logic |
| `pipeline_tools/prepare_transcript_jobs.py` | JSONL job bundle assembly for GCP Vertex AI |

### `Michele_OneDrive/` — scripts, plots, small outputs, docs
| Included | Excluded |
|----------|---------|
| All `.py` scripts (RFEC build/classify/cluster/plot/semantics) | Large computed CSVs > 1 MB (see below) |
| All `.png` plots (all < 1 MB) | |
| Small summary CSVs (centroids, thresholds, counts, shares — all < 0.2 MB) | |
| `ClusterYT.xlsx` (0.27 MB) | |
| `CamiHawke context description.docx` | |
| `camihawke_community_personas_refined.md` | |
| `camihawke_persona_coding_grid.md` | |
| `RFEC/RFEC Ig pipeline.md` | |

---

## EXCLUDED from git (covered by .gitignore)

### Binary data files
- `*.parquet` — all platforms' cleaned datasets (~600 MB total)
- `*.pkl` — Facebook and TikTok pickled data (~74 MB)

### Transcript raw files
- `transcripts/` — 10,070 files (~5–10 GB): VTT subtitles, `.empty` and `.failed` markers
- Stray `*.it.vtt` at root level

### GCP job bundles
- `Data_Cleaned/gcp_jobs/` — production JSONL (~428 MB) and test variants (~1 GB total)

### Pipeline runtime artifacts
- `Data_Cleaned/completed_video_ids.txt`
- `Data_Cleaned/processed_urls.txt`
- `Data_Cleaned/pooled_*.csv`
- `Data_Cleaned/cloud_matched_transcripts.csv`
- `Data_Cleaned/pooled_transcript_summary.txt`
- `Data_Cleaned/recycle_manifest*.json`

### Auth / sensitive
- `*_cookies.txt` (YouTube session cookies)

### Michele large computed CSVs (> 1 MB, regenerable from scripts)
- `ig_dynamic_rfe_transitions_long.csv` (77 MB)
- `ig_users_dynamic_rfe_clusters_long_all_windows.csv` (62 MB)
- `ig_comments_dynamic_rfe_cluster_assignment_long.csv` (37 MB)
- `ig_comments_dynamic_rfe_level_classification_long.csv` (28 MB)
- `ig_comments_dynamic_rfe_rolling_metrics.csv` (20 MB)
- `ig_comments_dynamic_rfe_input.csv` (14 MB)
- `ig_users_dynamic_rfe_clusters_wide.csv` (5 MB)
- `fb_comments_RFE_macro_top_level_annual_*.csv` (1.5 MB)

### Python cache
- `__pycache__/`, `*.pyc`

---

## Estimated git repo size after filtering
~500 KB (code + docs + small CSVs + PNGs)
vs ~12 GB total workspace