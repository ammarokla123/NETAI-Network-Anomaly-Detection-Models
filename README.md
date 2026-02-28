# 🔍 NETAI: Network Anomaly Detection Models

> Deep learning models for automated network anomaly detection using perfSONAR telemetry — a GSoC 2026 prototype for the [National Research Platform (NRP)](https://nrp.ai/).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen.svg)](#testing)

---

## Overview

This project implements **reconstruction-based anomaly detection** for network telemetry from the National Research Platform. It automatically identifies:

- 🐌 **Slow links** — degraded throughput with elevated latency
- 📉 **High packet loss** — excessive packet loss percentages
- 🔄 **Excessive retransmits** — abnormal TCP retransmission counts
- ❌ **Failed tests** — complete test failures (zero throughput)
- 📊 **High jitter** — unstable latency patterns

Three model architectures are implemented, trained, and evaluated:

| Model | Architecture | Parameters | ROC-AUC | PR-AUC |
|-------|-------------|-----------|---------|--------|
| **Autoencoder** | Fully-connected encoder-decoder | 13,796 | 0.820 | 0.216 |
| **LSTM** | Bidirectional LSTM with attention | 176,125 | 0.932 | 0.990 |
| **Transformer** | Multi-head self-attention encoder | 120,380 | 0.945 | 0.992 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETAI Anomaly Detection                       │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer │   Models     │  Training    │  Serving            │
│             │              │              │                     │
│ ┌─────────┐ │ ┌──────────┐ │ ┌──────────┐ │ ┌───────────────┐  │
│ │SQLite DB│ │ │Autoencoder│ │ │ Trainer  │ │ │ FastAPI REST  │  │
│ │(perfSON.)│ │ │          │ │ │          │ │ │  /predict     │  │
│ └────┬────┘ │ ├──────────┤ │ ├──────────┤ │ │  /predict/bat.│  │
│      │      │ │  LSTM    │ │ │Checkpoint│ │ │  /health      │  │
│ ┌────▼────┐ │ │(BiLSTM)  │ │ │  Mgmt    │ │ │  /model/info  │  │
│ │Feature  │ │ ├──────────┤ │ ├──────────┤ │ └───────┬───────┘  │
│ │Pipeline │ │ │Transform.│ │ │ Early    │ │         │          │
│ │(rolling,│ │ │(Attn.)   │ │ │ Stopping │ │ ┌───────▼───────┐  │
│ │ lag,    │ │ └──────────┘ │ └──────────┘ │ │  Kubernetes   │  │
│ │ diff,   │ │              │              │ │  Deployment   │  │
│ │ scale)  │ │              │              │ │  (GPU pods)   │  │
│ └─────────┘ │              │              │ └───────────────┘  │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

## Project Structure

```
├── src/netai_anomaly/
│   ├── data/
│   │   ├── schema.py          # SQLite schema (perfSONAR-style tables)
│   │   ├── generator.py       # Synthetic telemetry data generator
│   │   ├── features.py        # Feature engineering pipeline
│   │   └── dataset.py         # PyTorch Dataset classes
│   ├── models/
│   │   ├── base.py            # Base model + registry
│   │   ├── autoencoder.py     # FC Autoencoder
│   │   ├── lstm.py            # BiLSTM with temporal attention
│   │   └── transformer.py     # Transformer encoder
│   ├── training/
│   │   ├── trainer.py         # Training loop with early stopping
│   │   └── utils.py           # Seed management
│   ├── evaluation/
│   │   ├── metrics.py         # Precision, Recall, F1, ROC-AUC, PR-AUC
│   │   └── visualize.py       # Training curves, ROC, PR, score plots
│   └── inference/
│       └── service.py         # FastAPI REST inference service
├── scripts/
│   ├── generate_data.py       # Data generation CLI
│   ├── train.py               # Model training CLI
│   ├── evaluate.py            # Evaluation & plotting CLI
│   └── serve.py               # Inference server CLI
├── configs/                   # YAML configuration files
├── tests/                     # 56 comprehensive tests
├── k8s/                       # Kubernetes manifests
├── Dockerfile                 # Inference container
├── Dockerfile.training        # GPU training container
└── docker-compose.yaml        # Local development
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NETAI-Network-Anomaly-Detection-Models.git
cd NETAI-Network-Anomaly-Detection-Models

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[dev,plots]"
```

### 2. Generate Synthetic Data

```bash
python scripts/generate_data.py --num-samples 50000 --anomaly-ratio 0.05
```

This creates a SQLite database at `data/network_telemetry.db` with realistic perfSONAR-style measurements including throughput, latency, packet loss, retransmits, and jitter.

### 3. Train a Model

```bash
# Train any of the three architectures
python scripts/train.py --model autoencoder --epochs 50
python scripts/train.py --model lstm --epochs 30
python scripts/train.py --model transformer --epochs 30

# With GPU acceleration
python scripts/train.py --model transformer --device cuda
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/transformer_best.pt
```

Generates evaluation metrics and plots in `outputs/<model>/`.

### 5. Serve the Model

```bash
python scripts/serve.py --checkpoint checkpoints/transformer_best.pt --port 8000
```

Then query the API:

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "throughput_mbps": 500.0,
    "latency_ms": 200.0,
    "packet_loss_pct": 15.0,
    "retransmits": 150,
    "jitter_ms": 45.0
  }'

# Response:
# {"is_anomaly": true, "anomaly_score": 0.523, "threshold": 0.154, "confidence": 0.87}

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"throughput_mbps": 9500, "latency_ms": 5, "packet_loss_pct": 0.01, "retransmits": 2, "jitter_ms": 0.5}]}'
```

## Testing

```bash
# Run all 56 tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=netai_anomaly --cov-report=term-missing
```

Test coverage includes:
- **Data layer**: SQLite schema, data generation, reproducibility, roundtrip I/O
- **Feature engineering**: Rolling stats, lag features, normalization, pipeline fit/transform
- **Models**: Forward pass shapes, anomaly scores, gradient flow, all architectures
- **Training**: Loss convergence, checkpointing, threshold computation, early stopping
- **Inference API**: All endpoints, error handling, batch processing, validation

## Kubernetes Deployment

### Inference Service

```bash
# Build and deploy
docker build -t netai-anomaly:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment-inference.yaml
kubectl apply -f k8s/service-inference.yaml
```

### GPU Training Job

```bash
# Build training image and launch on NRP GPU cluster
docker build -f Dockerfile.training -t netai-anomaly-training:latest .
kubectl apply -f k8s/job-training.yaml
```

The training job requests NVIDIA GPU resources and includes proper tolerations for GPU-enabled nodes on the NRP Kubernetes cluster.

## Feature Engineering Pipeline

The pipeline transforms raw telemetry into model-ready features:

1. **Rolling statistics** — Mean and standard deviation over windows of 5, 15, and 30 time steps
2. **Lag features** — Previous values at lags of 1, 3, 5, and 10 steps
3. **Rate of change** — First-order differences for trend detection
4. **Normalization** — StandardScaler, MinMaxScaler, or RobustScaler

Starting from 5 raw metrics, the pipeline produces **60 engineered features** per sample.

## Model Details

### Autoencoder
Fully-connected encoder-decoder network that compresses telemetry into a low-dimensional latent space. Anomalies produce high reconstruction error because the model has only learned to reconstruct normal patterns.

### LSTM (Long Short-Term Memory)
Bidirectional LSTM with temporal attention that captures sequential dependencies in network time series. Processes sliding windows of 60 time steps to detect anomalous temporal patterns.

### Transformer
Multi-head self-attention encoder with sinusoidal positional encoding. Excels at capturing long-range dependencies and achieves the highest ROC-AUC (0.945) among all models.

## Configuration

All hyperparameters are managed through YAML files in `configs/`:

```yaml
# configs/default.yaml
data:
  sequence_length: 60
  anomaly_ratio: 0.05
feature_engineering:
  rolling_windows: [5, 15, 30]
  lag_steps: [1, 3, 5, 10]
  scaler: "standard"
training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  patience: 10
  scheduler: "cosine"
```

## Technologies

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, Autoencoder, LSTM, Transformer |
| **ML/Data** | scikit-learn, Pandas, NumPy |
| **Storage** | SQLite (perfSONAR telemetry) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Infrastructure** | Docker, Kubernetes, GPU Pods |
| **Testing** | pytest (56 tests), pytest-cov |
| **Config** | YAML, argparse |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- [National Research Platform (NRP)](https://nrp.ai/) for infrastructure and LLM/GPU services
- [perfSONAR](https://www.perfsonar.net/) for network measurement tools
- [ESnet](https://www.es.net/) for network monitoring tooling
- Mentors: Dmitry Mishin, Derek Weitzel
