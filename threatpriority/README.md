# ThreatPriority - Real-Time Cyber Threat Risk Scoring

MLCS-HACK-07 | Machine Learning for Cybersecurity | II Year Cybersecurity

A local web application that scores incoming security events, explains scores with genuine Tree SHAP, and helps an analyst investigate the highest-scoring alerts first. It works without paid APIs or cloud hosting.

**Start with the bundled synthetic dataset.** This release contains no Kaggle or captured production traffic. Synthetic/public logs are permitted by the problem statement. The generator, scenario assumptions, labels, seeds, and limitations are documented. Compatible Kaggle CSVs can be uploaded for scoring; training on a different dataset requires the workflow in `KAGGLE_GUIDE.md`.

## Windows (Python 3.12.10, 64-bit)

Install Python 3.12.10 and select Add Python to PATH. Extract this ZIP, open the `threatpriority-shap` folder in VS Code, and open a **Command Prompt** terminal there:

```bat
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py --port 5050
```

The first command should show Python 3.12.10. Using the environment's executable directly avoids PowerShell activation-policy issues. These executable commands also work in PowerShell. Keep the terminal open while using the app.

Alternatively, double-click `setup_windows.bat` once, then `run_windows.bat` for each run.

## Mac (including M4 / Apple Silicon)

Use native **Python 3.12**, preferably 3.12.10, and open a terminal in this folder:

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py --port 5050
```

If `python3.12` is unavailable but `python3 --version` shows 3.12.x, substitute `python3` for the first two commands. Keep the architecture consistent: use native arm64 Python on Apple Silicon, rather than mixing Intel and arm64 packages.

Alternatively run `bash setup_mac.sh` once, then `bash run_mac.sh`.

## Open the dashboard

Visit **http://127.0.0.1:5050** in Chrome, Edge, or Safari. Click **Load synthetic demo**, select an alert, and inspect its SHAP evidence. Press Ctrl+C in the terminal to stop the server. On future runs, use only the final `app.py` command. Internet is needed for the first dependency installation; subsequent app use works offline. The model is already trained.

## Demonstration sequence

1. Load the synthetic demo: a 300-event ranked queue appears.
2. Set the investigation budget K to 20 and compare ML priority with arrival order.
3. Select an alert. Positive SHAP contributions raise its model score; negative contributions lower it.
4. Upload `data/demo_missing_columns.csv`. Missing fields are imputed and prominently flagged.
5. Upload `data/demo_unlabeled.csv`. Scoring still works; quality evaluation is unavailable without labels.
6. Upload `data/held_out_test.csv` for the held-out experiment. This file was not used to train or choose the model.
7. Start recorded replay: one synthetic record is ingested through the API each second.
8. Record an analyst decision and export the ranked CSV. Feedback does not change scores or ground-truth labels.

Uploads replace the active queue; API/manual records append. Replay starts a fresh queue. The active queue is in memory and resets on restart. Feedback history persists in `runtime/feedback.sqlite`; clearing the queue does not delete that history. The app supports 1-10,000 rows per upload and an 8 MB request limit, and is intended for one local user.

## Output and score definition

Each event has a 0-100 score, severity, priority, observed network category (service/protocol), timestamp where available, evidence completeness, and review state. Category is **observed service/protocol**, not a predicted attack-family classification.

`risk_score = 100 * RandomForest.predict_proba(processed_event)[attack]`

This is an **uncalibrated attack-likelihood proxy**, not a measured business-loss risk or guaranteed probability of an actual attack. Fixed operational score bands assign priorities: Low/P4 below 35, Medium/P3 from 35 to below 65, High/P2 from 65 to below 85, Critical/P1 from 85 upward. Detection metrics use a separate, fixed threshold of 50. These thresholds are project policy, not official SOC standards.

## Missing-data contract

At least 3 of the 9 recognized features must be usable in each event. Absent numeric columns and invalid/negative numeric values are imputed using training medians. Numeric missing indicators are included in the model. Missing categories receive a dedicated token; unknown categories are encoded without a matching one-hot value and trigger a warning. Imputed contributions are labeled as assumptions in the SHAP panel. No arbitrary missing field is treated as zero observed activity. Every row must satisfy validation; invalid batches are rejected atomically without replacing the current queue.

No timestamp is fabricated. Timestamps accept ISO strings or Unix **seconds**. If any timestamp is invalid/absent, the arrival baseline uses CSV row/ingestion order for the entire queue. Labels are optional for scoring but required for evaluation. Unknown/missing labels disable evaluation rather than guessing them. See `DATA_DICTIONARY.md`.

## CSV evaluation

Precision, Recall, F1 and FPR use ground-truth labels and the threshold of 50. Precision@K, Recall@K and binary NDCG@K measure ranking. The app compares the top K ML events with K events in timestamp order (file order if timestamps are incomplete). K is capped at the number of events. Ties preserve input order. No-positive recall/ranking quantities and no-negative FPR are shown as N/A where undefined.

Average precision summarizes ranking over thresholds; Brier score measures squared error of probability estimates. Metrics from arbitrary uploaded training data are descriptive, not an independent performance claim. Labels never enter model features or queue ordering. Analyst feedback is not substituted for ground truth.

## Reproduce training and checks

Use the same pinned requirements to load the bundled model. Optional development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python train.py
python -m pytest -q tests/test_project.py
python generate_report.py
```

Run these with your environment's Python executable, or activate the environment first. `train.py` regenerates the synthetic data and overwrites the default model/report metrics. It compares Logistic Regression with Random Forest on validation data, then evaluates the fixed Random Forest on a separate test set. The split is 3,600 training, 1,200 validation and 1,200 test events. Feature-identical rows are removed before splitting. Training-only missingness augmentation is applied; preprocessing and SHAP background statistics use training data only.

For browser tests, install the development dependencies and run `python -m playwright install chromium`, start `app.py`, then `python tests/browser_check.py` in another terminal. Optional environment variables: `TP_BASE_URL` and `TP_BROWSER_EXECUTABLE`.

## Incoming-event API

Send JSON to `POST /api/events` with `Content-Type: application/json` and `X-Requested-With: ThreatPriority`. The header guards local browser writes. Example body:

```json
{"event":{"timestamp":"2026-09-30T10:00:00Z","duration":2,"src_bytes":8000,"dst_bytes":300,"src_packets":80,"dst_packets":4,"failed_logins":22,"unique_dst_ports":2,"protocol":"tcp","service":"ssh"}}
```

A batch uses `{"events":[{...},{...}]}`. GET `/api/queue` returns ranked outputs. The dashboard polls every two seconds for external incoming events. GET `/api/explain/<id>` returns SHAP contributions. GET `/api/evaluate?budget=20` returns evaluation JSON. GET `/api/export` downloads ranked CSV. API labels, if present, are evaluation-only. See `ARCHITECTURE.md` for the full flow.

## Practical troubleshooting

- `No module named ...`: install requirements using the same environment Python used to run `app.py`.
- Port 5050 busy: run `app.py --port 5052` and open that port's URL.
- Wrong Python version: recreate the environment with Python 3.12; Python 3.14 is not supported by this pinned stack.
- Model version warning: install the exact pinned packages; retrain with the same environment if needed. Never load model files from untrusted sources.
- CSV fails: save as UTF-8, use comma-separated headers, map compatible columns, convert units, and provide at least 3 usable features per row.
- Kaggle columns differ: read `KAGGLE_GUIDE.md`; renaming a header does not make different measurements equivalent.

## Evidence and limits

The bundled results are measured synthetic-data results, not production detection performance. Random splits do not prove chronological generalization; real logs need time/session/host-aware splitting. SHAP explains the fitted model and may divide credit between correlated measurements; it does not prove attack causation. Live network capture, automatic blocking, production SOC integration, online learning, user accounts, and deployment are outside this release.

Tests were run on Linux, including a clean Python **3.12.10** environment. Binary dependency resolution was checked for Windows x64 and Apple Silicon macOS Python 3.12. Native Windows/Mac execution was not available here; portable startup scripts and no OS-specific ML code are supplied. See `reports/VALIDATION.md` for the exact verification record.
