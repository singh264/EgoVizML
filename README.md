# EgoVizML

This repository contains all data processing code for the thesis:

> **"Using wearable technology to inform clinical decision-making in outpatient neurorehabilitation"**
> — Adesh Kadambi

---

## Repository Structure

```
EgoVizML/
├── egoviz/                    # Core Python package
│   ├── models/                # Feature processing, evaluation, inference
│   └── cdss_utils/            # Dashboard metrics, survey stats, video utilities
│
├── scripts/                   # Runnable pipeline scripts (see Pipeline section)
├── notebooks/                 # Analysis notebooks (numbered in run order)
├── data/                      # Processed prediction data (see data/README.md)
├── models/                    # Trained model artifacts
├── figures/                   # Output figures from analyses
├── shan_model/                # Vendored hand-object detector (external repo)
├── vizlabel/                  # Standalone labeling app
└── tests/                     # Test suite
```

External repositories that must be cloned separately (see [SETUP.md](SETUP.md)):

- `detectron2/` — Facebook's Detectron2 (required by Detic)
- `Detic/` — Open-vocabulary object detector

---

## ADL Recognition Pipeline

The pipeline takes raw egocentric video and produces per-video ADL predictions. It has five steps:

```
Raw video
    │
    ▼
[Step 1] Extract frames
    scripts/video_to_subclips_and_frames.py
    → 60-second subclips at 2 FPS, organized by patient/session
    │
    ▼
[Step 2a] Run Detic (object detection)
    scripts/run_detic.py
    → Per-frame object detections saved as .pkl files
    │
[Step 2b] Run SHAN (hand-object interaction detection)
    scripts/run_shan.py
    → Per-frame hand + contact-object detections saved as .pkl files
    │
    ▼
[Step 3a] Post-process Detic output
    scripts/process_detic.py
    → Maps 1000+ Detic classes → 29 functional categories
    → Filters human detections
    │
[Step 3b] Combine Detic + SHAN → active objects
    scripts/process_all_preds.py
    → Labels objects as "active" if IoU with hand box > 0.75
    → Output: home_data_all_preds.pkl
    │
    ▼
[Step 4] Feature generation + classification
    egoviz/models/processing.py + inference.py
    → Binary + Active features → Logistic Regression classifier
    → Best performance: mean F1 = 0.78, AUC = 0.94 (LOGOCV)
    │
    ▼
[Step 5] Dashboard metrics
    scripts/get_dashboard_metrics.py
    → Computes metrics for the clinical dashboard
```

### Expected folder structure for raw data

```
data_root/
├── communication-management/
│   └── <patient_id>/
│       ├── subclips/           ← output of Step 1
│       └── subclips_shan/      ← output of Step 2b
├── functional-mobility/
│   └── ...
└── ...other ADL folders...
```

### ADL Classes

Defined by the American Occupational Therapy Association (AOTA), Occupational Therapy Practice Framework (2020):

| Class                        | Description                                        |
| ---------------------------- | -------------------------------------------------- |
| `communication-management`   | Use of phones, computers, writing tools            |
| `functional-mobility`        | Moving from one position or place to another       |
| `grooming-health-management` | Hair, skin, oral care, medication routines         |
| `home-management`            | Maintaining household possessions and environment  |
| `meal-preparation-cleanup`   | Planning, preparing, and serving meals             |
| `self-feeding`               | Bringing food/fluid from plate or cup to mouth     |
| `leisure-other-activities`   | Non-obligatory, intrinsically motivated activities |

---

## Quick Start

See [SETUP.md](SETUP.md) for full environment setup instructions.

```bash
# 1. Install dependencies
pip install poetry
poetry install

# 2. Activate environment
poetry shell

# 3. Run tests to verify everything works
pytest tests/
```

### Running inference on new data

```python
from egoviz.models import processing, inference

# Load processed predictions (output of scripts/process_all_preds.py)
preds = processing.load_pickle("data/home_data_all_preds.pkl")

# Generate features
df = processing.generate_df_from_preds(preds)
df_features = processing.generate_binary_presence_df(df)
df_scaled = processing.row_wise_min_max_scaling(df_features)

# Load model and predict
model = inference.load_production_model("models/binary_active_logreg.joblib")
predictions = inference.predict(df_scaled, model)

# Access results
print(predictions.select(["predicted_label"]))
```

---

## Key Results

The best model configuration (Binary + Active features, Logistic Regression) achieved:

| Metric                  | Value |
| ----------------------- | ----- |
| Mean F1 (LOGOCV)        | 0.785 |
| Median F1 (LOGOCV)      | 0.812 |
| AUC                     | 0.94  |
| % subjects above 0.5 F1 | 100%  |

See `notebooks/13_FULL_ABLATION_STUDY.ipynb` for the full comparison across feature sets and classifiers.

---

## Notebooks

See [notebooks/README.md](notebooks/README.md) for a guide to the analysis notebooks.

---

## Dependency Management

```bash
# Add a dependency
poetry add <package>

# Remove a dependency
poetry remove <package>

# Update the egoviz package after changes
poetry install
```
