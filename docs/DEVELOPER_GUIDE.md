# MaintAI — Developer Guide

Technical reference for running, configuring, and extending the project locally or in Docker.

**System overview, architecture, and design rationale:** see the main [README](../Readme.md).

---

## Prerequisites

| Tool | Version / notes |
|------|-----------------|
| Python | 3.10+ recommended |
| Node.js | LTS (e.g. v20+) for frontend |
| Docker | Optional but recommended for full stack |
| RAM | 8 GB+ for large uploads (semantic + UMEC on 100k+ rows) |

---

## Quick start (Docker)

From the repository root:

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:5050 |
| Health check | http://localhost:5050/api/health |

Stop:

```bash
docker compose down
```

The backend container has an **8 GB memory limit** in `docker-compose.yml` to reduce OOM (exit 137) on large scoring runs.

---

## Local development (without Docker)

### Python environment

```bash
./scripts/install-dev.sh
source .venv/bin/activate   # macOS / Linux
```

`pyproject.toml` defines runtime dependencies; `requirements.txt` is a pinned dev snapshot.

### Backend

From project root (with venv active):

```bash
export PYTHONPATH=src
python backend/run.py
```

Runs on http://localhost:5000 (frontend dev proxy targets port **5050** when using Docker backend only).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on http://localhost:5173. Vite proxies `/api` to `VITE_API_PROXY_TARGET` (default `http://127.0.0.1:5050` on host, `http://backend:5000` in Compose).

### Production build (frontend)

```bash
cd frontend
npm run build
npm run preview
```

---

## Project layout

```text
configs/core/           project, data, and model YAML
configs/mappings/       failure keywords, part vocabulary, label maps
src/umec/               Python package (models, pipeline, data, XAI)
backend/                Flask API (routes thin; logic in app/services)
frontend/               React + Vite UI (shadcn, zinc dark theme)
scripts/                CLI entry points (train, predict, evaluate)
models/                 Saved .joblib artifacts (optional persistence)
docs/                   Documentation (this file)
reports/                CLI metrics and prediction exports (gitignored — local only)
data/raw/               Place CSV/Excel locally (gitignored — never commit)
data/processed/         Preprocessed outputs (gitignored)
history/                Run snapshots + feedback archives (gitignored)
unclassified/           Docker upload volume (gitignored)
```

**Data policy:** Customer maintenance records and exports must **not** be committed. Only mapping configs under `configs/mappings/` are versioned.

### Layering rules

| Layer | Responsibility |
|-------|----------------|
| `frontend/src` | UI only; all HTTP via `services/api.js` |
| `backend/app/routes` | HTTP, validation, JSON — no ML algorithms |
| `backend/app/services` | Orchestration, caching, jobs, calls into `umec` |
| `src/umec` | Models, preprocessing, pipeline, XAI |

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/core/project.yaml` | Paths, random seed, log level |
| `configs/core/data.yaml` | Data paths, text/label columns, preprocessing, resource JSON paths |
| `configs/core/model.yaml` | Base classifiers and UMEC (ECOC, spectral, prior) settings |

Common `data.yaml` fields:

- `source_text_column` — raw narrative in training CSV (e.g. `Discrepancy`)
- `text_column` — processed text used by models (e.g. `processed_discrepancy`)
- `label_column` — optional reference labels for evaluation only

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNCLASSIFIED_DIR` / `MAINTAINER_UPLOAD_DIR` | `/app/unclassified` | Uploads, predict job results |
| `FLASK_ENV` | — | Flask environment |
| `VITE_API_PROXY_TARGET` | `http://127.0.0.1:5050` | Vite → backend proxy |

---

## Web UI workflow

### Analysis workspace (`/`)

1. **Upload** CSV or Excel (stored server-side; large files return preview only).
2. **Define categories** (~10 failure-mechanism labels per site); keywords auto-generate from corpus. Presets save to browser `localStorage`.
3. **Map columns** — text (required), optional part/asset column, optional **reference label column** (enables evaluation on Results).
4. **Select models** — token, equipment, semantic, and/or UMEC ensemble.
5. **Run classification** — unsupervised fit on upload, then scoring (async job if &gt; 5k rows). A **run snapshot** is saved automatically on success.

Column/model selections and categories persist in **AppContext** and **sessionStorage** when you navigate away or refresh (predictions reload from the saved snapshot).

Optional: **Save models to disk** persists joblibs for CLI reuse or `use_saved_models` on the API.

### Results & review (`/results`)

1. Review all rows; filter by tier; edit **Final** labels.
2. **Bulk actions** — accept trusted rows (all or per label), revert edits, undo (see below).
3. **Classification report** — only if a reference label column was set at run time (metrics, confusion matrix, top confusion pairs).
4. **Active learning** — suggest/apply keywords from rows where Final ≠ Predicted.
5. **Export** — dataset sheet with ALL CAPS label column + separate findings sheet.

### Run history (`/history`)

- Lists saved runs (snapshots and legacy feedback records).
- **Open in Results** restores predictions, categories, and column mapping from the snapshot.

---

## Session persistence

| Storage | Key / path | Contents |
|---------|------------|----------|
| Browser | `sessionStorage` → `maintai_session_v1` | `runConfig`, `analysisConfig`, `uploadId`, `rowCount`, `columns`, preview rows (≤500), `lastRunId` |
| Server | `history/<timestamp>_<uuid>.json` | Full `snapshot` including `prediction` |

On app load (`AppProvider`):

1. Restore workspace fields from sessionStorage.
2. If `lastRunId` is set, fetch `GET /api/history/<id>` and restore `prediction` from `snapshot`.

**Limits:** Full row payloads are not stored in sessionStorage for large files. After a **backend restart**, `upload_id` may be invalid for server-side export — re-upload if export fails.

**Code:** `frontend/src/utils/sessionPersistence.js`, `frontend/src/context/AppContext.jsx`.

---

## Run snapshots (history)

### Save (automatic)

After each successful predict, the frontend calls:

```http
POST /api/history/snapshot
Content-Type: application/json

{
  "user": "anonymous",
  "snapshot": {
    "prediction": { ... },
    "runConfig": { "text_column", "label_column", "part_column", "models" },
    "analysisConfig": { "custom_categories", "user_settings" },
    "columns": ["..."],
    "uploadId": "...",
    "rowCount": 173000,
    "previewOnly": true
  }
}
```

Response: `{ "id", "timestamp", "model_version", "num_records", "type": "run_snapshot" }`.

### Restore

```http
GET /api/history/<record_id>
```

Returns the full record including `snapshot`. The UI uses `restoreRunById(id)` from `AppContext`.

**Backend:** `backend/app/services/umec_storage.py` → `save_run_snapshot()`.

---

## Bulk review API (client-side)

Bulk accept/revert runs entirely in the browser on `editedPredictions`. No dedicated API — use `POST /api/feedback` after review if you want an audit trail of final labels.

**UI component:** `frontend/src/components/ReviewBulkActions.jsx`.

| Action | Effect |
|--------|--------|
| Accept all trusted | `final_condition` ← `predicted_condition` for auto-accept rows |
| Accept trusted for label | Same, where `predicted_condition` matches |
| Revert all edits | All rows: `final` ← `predicted` |
| Revert for label | Rows tied to that label reset to predicted |
| Undo | Restores previous `editedPredictions` array (max 5 undo steps) |

---

## Evaluation and label drift

### When it appears

The classification report UI renders only when **`label_column`** was selected before predict (reference labels attached to each row as `actual_label`).

### API

```http
POST /api/evaluation/report
Content-Type: application/json

{
  "predictions": [ { "row_id", "actual_label", "predicted_condition", "final_condition", ... } ],
  "custom_categories": [ { "label": "leaking", "keywords": ["leak", "drip"] } ],
  "pred_key": "final_condition"
}
```

Response fields (subset):

| Field | Description |
|-------|-------------|
| `macro_f1`, `accuracy` | Scoped to your categories |
| `evaluated_rows`, `skipped_rows` | Mapped vs unmapped reference labels |
| `per_class` | Precision, recall, F1, support per label |
| `labels` | Category order for matrix |
| `confusion_matrix` | 2D counts (rows = actual, cols = predicted) |
| `top_confusion_pairs` | Off-diagonal pairs `[{ actual, predicted, count }]` |

**Code:** `src/umec/evaluation/scoped_report.py`, `backend/app/routes/evaluation.py`.

### CLI classification report

See [Classification report](#classification-report-after-dashboard-predict) below. Optional PNG:

```bash
python scripts/classification_report.py \
  --findings reports/findings.csv \
  --categories configs/my_site_categories.json \
  --confusion-matrix-png reports/confusion_matrix.png
```

---

## Active learning

Mines keywords from rows where engineers corrected the model prediction.

### API

```http
POST /api/feedback/active-learning
Content-Type: application/json

{
  "edits": [
    {
      "row_id": 1,
      "discrepancy": "hydraulic fluid dripping from fitting",
      "predicted_condition": "seeping",
      "final_condition": "leaking"
    }
  ],
  "custom_categories": [ { "label": "leaking", "keywords": ["leak"] }, ... ],
  "apply": false
}
```

| `apply` | Behaviour |
|---------|-----------|
| `false` | Returns `suggestions` map: label → list of new keyword tokens |
| `true` | Also returns `custom_categories` with keywords merged in |

**Code:** `backend/app/services/active_learning.py`, `src/umec/evaluation/category_matching.py`.

**UI:** Results page → **Active learning from your corrections** card. After apply, re-run classification on the workspace.

---

## API reference

Base path: `/api`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| POST | `/upload` | Parse CSV/Excel; returns `upload_id`, preview rows |
| POST | `/predict` | Fit + infer (`upload_id` or `rows`, `text_column`, `models[]`) |
| GET | `/predict/jobs/<id>` | Poll async job (`?result=1` for payload when done) |
| POST | `/train` | Persist models from JSON `dataset_meta` |
| POST | `/generate_tokens` | Auto keywords for label list + corpus |
| GET | `/models` | Model metadata for UI |
| GET | `/settings` | Analysis settings catalog |
| POST | `/feedback` | Save edited predictions (audit trail) |
| POST | `/feedback/active-learning` | Keyword suggestions from corrections (`apply` to merge) |
| GET | `/history` | List archived runs |
| GET | `/history/<id>` | Full record (includes `snapshot` for run_snapshot type) |
| POST | `/history/snapshot` | Save workspace + prediction snapshot |
| POST | `/export/labeled` | Server-side merge + download (CSV/XLSX) |
| POST | `/evaluation/report` | Scoped report + confusion matrix + top pairs |

### Predict (sync vs async)

- **≤ 5,000 rows:** synchronous `200` with full result JSON.
- **&gt; 5,000 rows:** `202` with `{ "async": true, "job_id": "..." }` — poll until `status: "done"`.
- Prefer **`upload_id`** for large files (avoids re-posting row payloads).

Example async flow:

```bash
# 1. Upload
curl -F "file=@data.csv" http://localhost:5050/api/upload

# 2. Start predict
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"upload_id":"<id>","text_column":"Discrepancy","models":["UMECClassifier"],"analysis_config":{"custom_categories":[{"label":"LEAKING","keywords":["leak","drip"]}]}}'

# 3. Poll
curl http://localhost:5050/api/predict/jobs/<job_id>
```

---

## CLI commands

Assume project root, venv active, `PYTHONPATH=src` (or editable install via `./scripts/install-dev.sh`).

### Train (config dataset)

```bash
python scripts/train.py --config configs/core
```

### Train + evaluate

```bash
python scripts/run_pipeline.py --config configs/core --evaluate
```

### Evaluate only

```bash
python scripts/evaluate.py --config configs/core
```

### Predict on a file

```bash
python scripts/predict.py \
  --config configs/core \
  --input data/raw/test.csv \
  --output reports/predictions.csv
```

With explainability:

```bash
python scripts/predict.py \
  --config configs/core \
  --input data/raw/test.csv \
  --output reports/predictions.csv \
  --include-xai \
  --top-k 3
```

### Explain one row

```bash
python scripts/explain.py --config configs/core --index 10
python scripts/explain.py --config configs/core --index 10 --use-saved
```

### Classification report (after dashboard predict)

Requires reference labels and your category list (same JSON as the workspace).

**Example categories file** (`configs/my_site_categories.json`):

```json
[
  { "label": "leaking", "keywords": ["leak", "drip", "hyd"] },
  { "label": "cracked", "keywords": ["crack", "fracture"] }
]
```

From **Findings** export CSV (after dashboard export):

```bash
python scripts/classification_report.py \
  --findings reports/findings.csv \
  --categories configs/my_site_categories.json \
  --output reports/classification_report.csv
```

From a single file with true + predicted columns:

```bash
python scripts/classification_report.py \
  --input data/eval.csv \
  --label-column PartCondition \
  --pred-column FAILURE_MECHANISM \
  --categories configs/my_site_categories.json
```

From saved predict API JSON:

```bash
python scripts/classification_report.py \
  --predictions-json reports/last_run.json \
  --model UMECClassifier \
  --categories configs/my_site_categories.json \
  --confusion-matrix-png reports/confusion_matrix.png
```

The report uses **only your defined categories**; CMMS reference values that do not map are skipped.

**Script:** `scripts/classification_report.py`

### Docker one-liner train

```bash
docker compose exec backend python scripts/train.py --config configs/core
```

---

## Large-file behaviour

| Threshold | Behaviour |
|-----------|-----------|
| &gt; 5k rows (API) | Async predict job; poll for progress |
| &gt; 5k rows (upload) | Preview only in browser (first 200 rows); full file on server |
| &gt; 10k rows (fit) | Large-dataset mode: sampled FastText (~2.5k), UMEC fit sample (~8k), predict chunks of 4k |

Watch backend logs for chunk lines, e.g. `UMEC ensemble: scoring chunk 12/45 (rows 44,000–48,000)`.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `network connection was lost` on predict | Proxy timeout or backend OOM | Use async path; keep tab open; check `docker compose logs -f backend` |
| Exit code **137** | OOM during scoring | Increase Docker memory; reduce models; ensure large-dataset mode active |
| Slow first predict | On-the-fly fit + FastText | Expected; repeat runs cache in memory until data/categories change |
| Upload fails | Backend down | Check http://localhost:5050/api/health |
| Macro F1 looks wrong | F1 computed over all CMMS codes | UI scopes F1 to **user categories** only when reference column set |
| Columns reset after navigation | Was local Dashboard state | Now in AppContext + sessionStorage; pull latest frontend |
| Reopen run fails | Snapshot missing or backend restarted | Re-run classification; ensure `history/` volume persists in Docker |
| sessionStorage quota | Very large preview in browser | Only ≤500 rows stored; predictions always from server snapshot |

### Performance tuning

- Semantic similarity is the slowest base model on large uploads.
- Adjust `semantic_similarity.workers` / `n_jobs` in `configs/core/model.yaml` (2–4 on 8 GB RAM).
- Token-only runs skip FastText entirely (seconds vs many minutes).

---

## Frontend development

- **Stack:** React 19, Vite, React Router, Tailwind 3, shadcn/ui (zinc dark).
- **Path alias:** `@/` → `frontend/src/`.
- **API client:** `frontend/src/services/api.js` — do not scatter `fetch` calls.
- **Global state:** `frontend/src/context/AppContext.jsx` — upload, prediction, `runConfig`, categories, `restoreRunById`.
- **Session:** `frontend/src/utils/sessionPersistence.js`.

### Key UI components

| Component | Purpose |
|-----------|---------|
| `ClassificationReportPanel.jsx` | Metrics table + confusion matrix + drift pairs |
| `ConfusionMatrixHeatmap.jsx` | Scrollable heatmap (many classes) |
| `ConfusionPairsPanel.jsx` | Top off-diagonal confusions |
| `ReviewBulkActions.jsx` | Bulk accept / revert / undo |
| `ActiveLearningPanel.jsx` | Keyword suggest / apply |
| `PredictionTable.jsx` | Row-level review and edit |
| `LoadingOverlay.jsx` | Progress + chunk bar for async jobs |

### Pages

| Route | File |
|-------|------|
| `/` | `pages/Dashboard.jsx` |
| `/results` | `pages/Results.jsx` |
| `/history` | `pages/History.jsx` |
| `/health` | `pages/Health.jsx` |

---

## Adding a new API route

1. Create `backend/app/routes/<name>.py` with a Blueprint.
2. Register in `backend/app/__init__.py`.
3. Put business logic in `backend/app/services/`.
4. Add a function in `frontend/src/services/api.js`.

---

## Paper ↔ code map

| Paper step | Code |
|------------|------|
| Token matching | `src/umec/models/token_matching.py` |
| Equipment-based | `src/umec/models/equipment_based.py` |
| Semantic / CWEM-style | `src/umec/models/semantic_similarity.py` |
| ECOC + max reduction | `UMECClassifier._reduction_stats` |
| Spectral moments | `umec.models.spectral` |
| Imbalance-aware decode | `UMECClassifier.class_score_df` + `prior_weight` |

---

## License

Internal research / capstone use unless otherwise agreed with industry partners.
