---
name: colab-ready
description: Transform a locally-developed Python script or Jupyter notebook into a sequential, headless Google Colab Enterprise notebook for scheduled execution. Use when the user wants to "colabify", deploy to Colab Enterprise, prepare code for a scheduled/headless cloud run, or convert local file paths + auth to GCS + default credentials.
---

# Colab Enterprise Ready

Refactor local code into a notebook that runs **unattended** on Colab Enterprise
(scheduled executor, headless, no human in the loop). The guiding principle:
**no local assumptions, no interactivity, every output persisted to GCS, every
failure logged loudly.**

## When to use
- User says: "make this run on Colab", "colabify", "prepare for scheduled run",
  "headless cloud version", "deploy this notebook to Colab Enterprise".
- After local code is verified working and needs to be promoted to the cloud.

## Output format
Produce a **sequential notebook** (cells run top-to-bottom, no out-of-order
dependencies). If editing an `.ipynb`, use NotebookEdit. If writing a plan/script,
clearly mark `# === CELL N ===` breaks. Keep the original logic intact — only
change what the checklist below requires.

## The transformation checklist

### 1. Dependencies (Cell 1)
- One `!pip install -q <pkgs>` cell at the very top for **non-standard** libs only.
- Do **not** list builtins (os, json, re, asyncio, pathlib, subprocess, logging…).
- Common cloud deps to watch for: `gcsfs`, `google-cloud-storage`,
  `google-cloud-aiplatform`, `pyarrow`, `xgboost`, `playwright` (+ a separate
  `!playwright install --with-deps chromium` cell if browser automation is used).
- Pin versions only if the code depends on specific APIs.

### 2. Config / placeholders (Cell 2)
- Hoist `PROJECT_ID`, `BUCKET_NAME` (and `LOCATION` if Vertex/AI is used) to the
  top as plain variables. Keep real values if already known in the project
  (this repo uses `GCP_PROJECT_ID = "gen-lang-client-0792749758"`,
  `GCP_BUCKET = "afb_showreel"`, `GCP_LOCATION = "us-central1"`).
- All paths become `gs://{BUCKET_NAME}/...` strings derived from these.

### 3. Storage I/O — everything through GCS
- Replace local read/write paths with `gs://{BUCKET}/{path}`.
- `pandas` reads/writes GCS natively when `gcsfs` is installed:
  `pd.read_parquet("gs://...")`, `df.to_csv("gs://...")`.
- For non-pandas artifacts (JSON, models, browser screenshots), upload with the
  `google.cloud.storage` client (`bucket.blob(path).upload_from_filename(...)`).
- **Colab disk is ephemeral**: anything written only to `./` is lost when the
  scheduled VM tears down. Every result the job is supposed to produce MUST end
  up in GCS before the notebook finishes.
- It is fine to stage to local disk for speed (e.g. `./work/`) as long as a final
  step uploads to GCS. Preserve any existing GCS load/save cells — don't drop them.

### 4. Authentication — implicit only
- **Remove** explicit credential loading: no `service_account.json`,
  no `GOOGLE_APPLICATION_CREDENTIALS` juggling, no `os.environ[...]` key reads,
  no `google.colab.auth.authenticate_user()` interactive prompts.
- Rely on **Application Default Credentials** — Colab Enterprise runs as a service
  account. Just `storage.Client(project=PROJECT_ID)` /
  `vertexai.init(project=..., location=...)` and let ADC resolve.

### 5. Execution flow — non-interactive
- Remove `input()`, `getpass()`, `plt.show()`, `display()`-only side effects,
  `tqdm` interactive widgets, `!` shell prompts that wait for input.
- Replace `print()` with the `logging` module so the scheduler captures clean,
  timestamped logs (see snippet). Plots: `plt.savefig("gs://..."?)` — actually
  savefig can't write gs:// directly; save to `./fig.png` then upload the blob.
- Headless browsers (Playwright): always `headless=True`. On Colab Enterprise
  (Linux) the **async** Playwright API runs fine directly in the notebook — you do
  **not** need the Windows `subprocess` + `ProactorEventLoop` workaround that local
  Windows/Jupyter requires. Drop that workaround when moving off Windows.

### 6. Error handling — fail loud, exit graceful
- Wrap the core pipeline in `try/except`, log the full traceback, then
  `raise` (so the scheduler marks the run FAILED) — do not swallow fatal errors
  silently. A silent success on a broken run is the worst outcome for a scheduled job.
- For per-item loops (e.g. per-video scraping), catch and log per item so one bad
  record doesn't kill the batch, but still surface a summary count of failures.

### 7. Rate limiting / external services (if scraping or hitting APIs)
- Keep concurrency bounded and add jittered delays so the headless job doesn't get
  the service account's IP blocked. Expose knobs (`CONCURRENCY`, `MIN/MAX_DELAY`).
- Checkpoint long runs to GCS periodically so an interrupted scheduled job resumes
  rather than restarts.

## Reusable snippets

**Logging setup (replaces print, Cell ~2):**
```python
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("pipeline")
log.info("Pipeline started")
```

**Fatal-error guard around the main body (last code cell):**
```python
import traceback
try:
    main()                      # or inline the core orchestration
    log.info("Pipeline finished OK")
except Exception:
    log.error("FATAL — pipeline aborted:\n%s", traceback.format_exc())
    raise                       # re-raise so Colab Enterprise marks the run FAILED
```

**GCS upload of local artifacts (ephemeral-disk safeguard):**
```python
from google.cloud import storage
from pathlib import Path
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)
for f in Path("./outputs").glob("*"):
    if f.is_file():
        bucket.blob(f"outputs/{f.name}").upload_from_filename(str(f))
        log.info("uploaded gs://%s/outputs/%s", BUCKET_NAME, f.name)
```

**pandas direct GCS I/O (needs gcsfs):**
```python
df = pd.read_parquet(f"gs://{BUCKET_NAME}/in/data.parquet")
df.to_csv(f"gs://{BUCKET_NAME}/out/result.csv", index=False)
```

## Final pre-flight checklist (verify before handing back)
- [ ] Cell 1 installs only non-builtin deps; browser install cell present if needed.
- [ ] PROJECT_ID / BUCKET_NAME / LOCATION hoisted to a config cell.
- [ ] No local-only paths remain; every output lands in GCS.
- [ ] No explicit credentials / no interactive auth; ADC only.
- [ ] No input()/show()/interactive widgets; print → logging.
- [ ] Core logic wrapped in try/except that logs traceback and re-raises.
- [ ] Headless browser = True; Windows subprocess workaround removed.
- [ ] Cells run top-to-bottom with no out-of-order state.
