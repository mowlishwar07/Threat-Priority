# Validation record

## Verified behavior

- **38 automated tests passed** in a fresh Python **3.12.10** environment on Linux x86-64, installed using the project's runtime requirements plus pytest. The runtime dependency set works without development-only plotting/report packages.
- Tests cover missing columns, invalid numeric values, aliases/mappings, mixed epoch/ISO timestamps, duplicate headers, rejection of sparse/invalid batches, label/scenario/timestamp exclusion from predictions, real SHAP additive reconstruction, imputed explanation labels, stable ranking ties, timestamp/row-order baselines, one-class metric handling, score-band boundaries, CSV/API ingestion, feedback separation, export escaping, and request limits/validation.
- **13 browser workflow checks passed** against the running Python 3.12.10 server. See browser_checks.json. Workflows include custom-K evaluation, missing/unlabeled CSVs, explanations, feedback, replay, external incoming-event refresh, export, pagination/search, mobile overflow checks and absence of JavaScript page errors.
- Desktop and 390-pixel screenshots were inspected. The 200% root-font check tests horizontal overflow; it is not a full accessibility audit or browser-zoom test.
- `pip check`: no broken package requirements. Both Mac shell scripts pass shell syntax checks. A separate custom-CSV training smoke test successfully creates a model and reconstructs its SHAP probability; that smoke test is not an independent detection result.
- Binary dependency downloads resolved for **Windows x64 CPython 3.12** and **Apple Silicon macOS CPython 3.12**. This verifies wheel availability/resolution; it is not native OS execution. Startup scripts and paths are supplied for each platform.
- The PDF report was rendered and all seven pages visually inspected.

## ML experiment

Dataset: 6,000 generated events; 3,600 training, 1,200 validation, 1,200 test. Random Forest validation F1 73.1%; Logistic Regression validation F1 57.0%. No test-set threshold search was performed.

Held-out synthetic test: Precision **90.1%**, Recall **62.5%**, F1 **73.8%**, FPR **2.8%**. TP=218, FP=24, TN=827, FN=131. Average precision=0.8401; Brier score=0.1066. Threshold=50 risk points.

At K=20: ML finds **20** attacks, arrival order finds **7**; Precision@20=100% vs 35%; Recall@20=5.7% vs 2.0%. These figures are specific to this synthetic split. See metrics.json, held_out_predictions.csv and ranking_budgets.json for auditable results.

Dropping dst_bytes, dst_packets and failed_logins from the same test file reduces F1 to **51.5%** and recall to **36.1%**, while scoring remains functional. Missing-column support does not mean absent evidence has no performance cost. This fixed-model stress test was not used to tune the model; see missing_columns_evaluation.json.

Warm single-event and SHAP timings are in latency.json. They are local machine measurements, not guaranteed throughput. Recorded replay is one event per second; the browser polls externally ingested events every two seconds.

## Verification limits

Native Windows and Mac sessions were unavailable. No production SOC integration, true live packet capture, chronological deployment experiment, or public/Kaggle benchmark is bundled. The random synthetic split measures the same generator distribution. Uploaded compatible files can be scored, but reliable cross-dataset performance requires provenance checks and independent validation. Analyst feedback is recorded without automatic retraining or alteration of ground truth.
