# ABD-GAF Credit Card Fraud Detection

This project implements **ABD-GAF (Adaptive Behavioral-Drift and Graph-Anomaly Fusion)** end-to-end on the Kaggle Credit Card Fraud Detection dataset.

## Project structure

- `data/creditcard.csv` - Kaggle dataset input file
- `src/preprocessing.py` - split, scaling, SMOTE pipeline
- `src/drift_module.py` - behavioral drift detector
- `src/graph_module.py` - kNN graph + GraphSAGE model
- `src/fusion.py` - drift/graph score fusion + threshold tuning
- `src/baselines.py` - Logistic Regression, Random Forest, XGBoost, LSTM baselines
- `src/evaluate.py` - metrics and graph generation
- `src/predict.py` - single and batch offline inference APIs
- `main.py` - orchestrates full training and benchmarking pipeline

## Setup

1. Place dataset at `data/creditcard.csv`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run full pipeline:

```bash
python main.py
```

## Outputs

- Trained models in `models/`
  - `logreg.pkl`, `rf.pkl`, `xgb.pkl`, `lstm.pt`, `gnn_best.pt`
  - `scaler.pkl`, `drift_detector.pkl`, `graph_context.pkl`, `fusion_config.pkl`
- Benchmark table in `results/benchmark_scores.csv`
- Plots in `results/graphs/`:
  - `01_roc_curves.png`
  - `02_pr_curves.png`
  - `03_confusion_matrix.png`
  - `04_f1_auprc_benchmark.png`
  - `05_drift_score_distribution.png`
  - `06_graph_score_distribution.png`
  - `07_fusion_score_density.png`
  - `08_gnn_training_loss.png`
- Batch inference output in `results/batch_predictions.csv` (when using `predict_batch`)

## Notes

- Primary metric is **AUPRC**, per dataset recommendation.
- `Time` and `Amount` are scaled using a train-fitted scaler.
- `V1..V28` are kept as provided.
- SMOTE is applied only to training data for baseline model fitting.
- Validation/test splits are never oversampled.
