# Experiment Tracking & Model Governance

The EU-MDR Regulatory Copilot uses **MLflow** with a local SQLite backend to track training runs, hyperparameters, per-class metrics, and model promotion decisions.

### Storage Architecture
- **Tracking Store URI**: `sqlite:///mlflow.db`
- **Experiment Scope**: `eu-mdr-event-classification`
- **Artifacts**: Stored in `models/` and registered into the active MLflow run.

### Launching the MLflow Dashboard
To launch the UI on Windows without multiprocess socket conflicts:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 127.0.0.1 --workers 1