# Model card

Purpose: prioritize candidate security events for human investigation within a fixed budget. Not an autonomous blocking system, production IDS, forensic conclusion, or incident-impact model.

Data: 6,000 synthetic events generated with seed 42; eight overlapping benign/attack scenarios, independent label noise and missingness. No Kaggle/public capture is bundled. Scenario assumptions are documented in DATA_DICTIONARY.md and ml/dataset.py. Nine allowlisted measurements form X; labels, scenarios, identifiers and timestamps are excluded.

Model: Random Forest, 120 trees, maximum depth 10, minimum 5 samples per leaf, fixed random seed. Training-only median imputation, numeric missing flags, and one-hot categorical encoding. Training-only missingness augmentation teaches partial-record behavior. Logistic Regression supplies a validation baseline. Forest selected for nonlinear modeling and native Tree SHAP explanation; it is not chosen by searching the held-out test.

Validation: after feature-identical duplicate removal, stratified random 60/20/20 split with seeds 17/18. Train 3,600; validation 1,200; test 1,200. Training augmentation is not counted as independent additional data. Statistics and SHAP background use training data only. Real-log validation should use independent collection periods/sessions/hosts as applicable.

Score: 100 times the forest's uncalibrated attack-class probability estimate. Priority bands are project policy: 35, 65, 85; detection metrics use 50. Severity is not a measured attack impact or a standard-defined incident category. Observed service/protocol forms the event category; there is no attack-family classifier.

Explanation: interventional Tree SHAP, probability output, 64 training-background rows. For binary model output, explain class 1. Group one-hot/missing-indicator contributions by original field; baseline + all contributions reconstruct the unrounded score. Missing inputs are labeled imputed. Correlated features and the chosen background affect attribution; SHAP describes the model rather than proving attack causation.

Measured held-out results are in metrics.json and Project_Report.pdf. Include recall, false negatives, and Brier score rather than relying only on top-K precision. Uploaded CSV evaluation measures that particular file, not independent validation unless the file genuinely is unseen and representative.

Missing-data limits: every event needs at least 3 usable schema features; missing fields are imputed and flagged. This minimum is a validation rule, not a confidence guarantee. Unknown categories warn; arbitrary schemas and distribution shift remain unsupported without redesign/retraining. Binary labels must be valid for every evaluated row. Missing capture times never receive invented dates.

Feedback: simulation stored in local SQLite. It does not alter model parameters, ground-truth labels, or evaluation outcomes. Runtime queue is transient and local. No production SOC integration or actual packet capture is claimed.
