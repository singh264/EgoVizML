# Analysis Notebooks

These notebooks document the full analysis journey for Aim 2 of the thesis. They are meant to be read and run in the order listed below. Each notebook has a header cell explaining what it does and what question it is answering.

All notebooks require the `egoviz` package to be installed (`poetry install`) and the Poetry kernel to be active in Jupyter.

---

## Core analysis sequence

| Notebook | Description |
|---|---|
| `00_scratch.ipynb` | Scratch space — ad hoc exploration, not part of the analysis sequence |
| `0_visualize_detections.ipynb` | Sanity check: visualize raw Detic + SHAN detections with bounding boxes on frames |
| `1_bag_of_objects.ipynb` | Explore the "bag of objects" representation — what objects appear in each ADL? |
| `2_adl_recognition.ipynb` | First attempt at ADL classification using Detic-only features |
| `3_adl_recognition_counts.ipynb` | ADL classification using count-based features |
| `4_effect_of_detections.ipynb` | Compare model performance on Detic detections vs. ground truth labels — does detection quality matter? |
| `hyperparameter_tuning.ipynb` | Hyperparameter search for Random Forest and Logistic Regression |
| `5_hyperparameter_tuning.ipynb` | Hyperparameter search for XGBoost |
| `6_effect_of_active_weights.ipynb` | Ablation: effect of weighting active objects more heavily |
| `7_effect_of_active_objects.ipynb` | Ablation: effect of including active object features |
| `8_best_active_weight.ipynb` | Find the optimal active object weight |
| `9_weighted_binary_presence.ipynb` | Evaluate weighted binary presence features |
| `10_binary_detections_active.ipynb` | Evaluate binary + active feature combination (the best configuration) |
| `11_computing_auc.ipynb` | Compute AUC for the best model configuration |
| `12_save_preds_and_figure.ipynb` | Save predictions and generate figures for the thesis |
| `13_FULL_ABLATION_STUDY.ipynb` | Full ablation: all feature set combinations × all classifiers |
| `14_SAVE_MODEL.ipynb` | Train the final production model on all data and save it to `models/` |

## Appendix

| Notebook | Description |
|---|---|
| `A1_01_stats_and_figures.ipynb` | Aim 1 analysis: descriptive statistics and figures from clinician survey results |

---

## Notes

- Notebooks were developed on Windows (Python 3.11) and may contain absolute paths like `C:\Users\adesh\...` in older cells. Update these to relative paths if re-running.
- The numbered sequence reflects the order of analysis, not necessarily the order things appear in the thesis.
- `hyperparameter_tuning.ipynb` (unnumbered) predates the numbered series and uses a simple train/test split rather than LOGOCV — treat it as exploratory.
