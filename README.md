# Student Anomaly Detection with MLOps

This project implements an anomaly detection model to identify students at risk of failure. It follows MLOps best practices including experimentation tracking, model versioning, and automated pipelines.

## 🚀 Features

- **Anomaly Detection**: Isolation Forest model to detect unusual student behavior.
- **Risk Assessment**: Hybrid scoring combining ML anomaly scores with rule-based heuristics.
- **Experiment Tracking**: Weights & Biases (W&B) integration for metric logging.
- **Model Registry**: Automatic versioning of trained models using W&B Artifacts.
- **CI/CD**: GitHub Actions pipeline for automated testing and checks.
- **API**: Flask-based REST API for real-time and batch predictions.

## 🛠️ Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd anomalydetectionmodel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases**
   - Create an account at [wandb.ai](https://wandb.ai)
   - Login locally:
     ```bash
     wandb login
     ```

## 🏃‍♂️ Training

To train the model and log experiments to W&B:

```bash
python train_model.py
```

This will:
- Load data from `data/`
- Train the Isolation Forest model
- Log metrics (F1 score, contamination) to W&B
- Save the model to `models/` and upload it to W&B Artifacts

## 🌐 API

Start the Flask API:

```bash
python app.py
```

Endpoints:
- `POST /predict`: Predict risk for a single student
- `POST /predict_batch`: Predict risk for a batch of students
- `GET /diagnose`: Run diagnostic tests

## 🧪 MLOps Pipeline

The project uses GitHub Actions for CI/CD:
- **Linting**: `flake8` checks for code style issues.
- **Testing**: `pytest` runs unit tests to ensure model integrity.
- **Trigger**: Runs on every push to `main` and pull requests.
