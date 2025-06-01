# Traffic4Cast - Temporal and Spatial Transfer Learning in Traffic Map Movie Forecasting
![image](https://github.com/user-attachments/assets/a6f3087d-9ab5-4afd-b712-ac31a049f90c)

This repository implements state-of-the-art transfer learning approaches for the Traffic4Cast competition, featuring novel spatio-temporal transfer methods using multi-task learning and few-shot adaptation that can predict COVID-era traffic patterns in unseen cities using only pre-COVID adaptation data.

## Key Features

- **Enhanced Spatio-Temporal Transfer Learning**: True transfer across both space (cities) and time (pre-COVID → COVID)
- **Multi-Task Learning Framework**: Joint training on traffic prediction, city classification, and temporal pattern recognition
- **Few-Shot Adaptation**: Meta-learning approach for rapid adaptation to new cities with minimal data
- **Comprehensive Metrics**: Including competition-specific MSE Wiedemann alongside standard metrics
- **Advanced Training Pipeline**: TQDM progress bars, Wandb integration, enhanced checkpointing
- **Flexible Experiment Configuration**: YAML-based config system supporting multiple experiment types

## Research Contributions

### 1. Enhanced Spatio-Temporal Transfer Learning
- **Problem**: Can a model trained on multiple cities predict traffic in a completely unseen city during COVID, given only that city's pre-COVID data?
- **Solution**: Multi-task learning with spatial attention, temporal patterns, and meta-learning adaptation
- **Innovation**: First true spatio-temporal transfer learning approach for traffic prediction

### 2. Multi-Task Learning Architecture
```python
# Multi-task UNet with auxiliary tasks
outputs = model(inputs, metadata, mode="train")
# Primary: traffic prediction
# Auxiliary: city classification, year classification  
# Enhanced: attention-based traffic prediction
```

### 3. Competition-Grade Metrics
- **MSE Wiedemann**: Official Traffic4Cast competition metric
- **Volume/Speed Decomposition**: Separate metrics for traffic volume and speed accuracy
- **Transfer Score**: Comprehensive evaluation metric for transfer learning quality

## 🛠️ Installation

### Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/traffic4cast-enhanced.git
cd traffic4cast-enhanced

# Create conda environment
conda env create -f environment.yaml
conda activate t4c

# Install additional dependencies
pip install wandb  # Optional: for experiment tracking
```

### Data Setup
1. Download Traffic4Cast dataset from the [official competition page](https://www.iarai.ac.at/traffic4cast/)
2. Extract data to `data/DATASET/` directory
3. Update data paths in configuration files

```
data/DATASET/
├── BARCELONA/
│   └── training/
│       ├── 2019-01-01_BARCELONA_8ch.h5
│       └── ...
├── MELBOURNE/
└── ...
```

## Quick Start

### Basic Spatial Transfer Learning
```bash
# Train on multiple cities, test on unseen city
python scripts/train.py --config config/spatial_config.yaml
```

### Enhanced Spatio-Temporal Transfer Learning
```bash
# Our main contribution: spatio-temporal transfer with adaptation
python scripts/train.py --config config/spatiotemporal_config.yaml
```

### Debug Mode
```bash
# Quick test with minimal data
python scripts/train.py --config config/debug.yaml --debug
```

## 📋 Experiment Types

### 1. Spatial Transfer Learning
Train on cities A, B, C → Test on city D
```yaml
experiment:
  type: "spatial_transfer"
  train_cities: ["ANTWERP", "BANGKOK", "BARCELONA"]
  test_cities: ["MELBOURNE"]
```

### 2. Enhanced Spatio-Temporal Transfer Learning
Train on multiple cities (2019+2020) → Adapt to new city (2019) → Test on new city (2020)
```yaml
experiment:
  type: "enhanced_spatiotemporal_transfer"
  train_cities: ["ANTWERP", "BANGKOK"]
  train_years: ["2019", "2020"]
  test_city: "BARCELONA"
  test_train_year: "2019"    # Adaptation data
  test_target_year: "2020"   # Target predictions
  adaptation_samples: 100
```

## Architecture Overview

### Multi-Task UNet
- **Shared Encoder**: Learns general traffic representations
- **Traffic Head**: Primary traffic prediction task
- **City Classifier**: Auxiliary task for spatial pattern learning
- **Year Classifier**: Auxiliary task for temporal pattern learning
- **Attention Modules**: Spatial and temporal attention for adaptive feature fusion

### Training Pipeline
1. **Multi-task Training**: Joint training on multiple objectives
2. **Few-shot Adaptation**: Meta-learning on target city's source year
3. **Transfer Evaluation**: Test on target city's target year

## Results & Metrics

### Key Metrics Tracked
- **MSE**: Standard mean squared error
- **MSE Wiedemann**: Competition-specific weighted MSE
- **Volume Accuracy**: Binary classification accuracy for traffic presence
- **Speed Accuracy**: Speed prediction accuracy where traffic exists
- **Transfer Score**: Composite metric for transfer learning quality

### Results
```
Enhanced Spatio-Temporal Transfer Results:
  Training Cities: ANTWERP, BANGKOK (2019+2020)
  Target City: BARCELONA (2019→2020)
  
  Metrics:
    Transfer Score: 0.4341
    Volume Accuracy: 0.6567
    Speed Accuracy: 0.5234
    MSE Wiedemann: 0.4876
```

## Configuration System

### Flexible YAML Configuration
```yaml
# config/spatiotemporal_config.yaml
model:
  name: "multitask_unet"
  use_attention: true
  use_meta_learning: true

training:
  batch_size: 4
  learning_rate: 0.0001
  traffic_weight: 1.0      # Primary task
  city_weight: 0.1         # Spatial auxiliary task
  year_weight: 0.1         # Temporal auxiliary task

experiment:
  type: "enhanced_spatiotemporal_transfer"
  # ... experiment configuration
```

### Command Line Overrides
```bash
# Override any config parameter
python scripts/train.py \
  --config config/base.yaml \
  --experiment-name my_experiment \
  --device cuda \
  --test-city MELBOURNE
```

## Testing & Validation

### Run Test Suite
```bash
# Test data loading and model architectures
python -m pytest tests/ -v

# Test MSE computation specifically
python tests/test_mse_metrics.py
```

### Model Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py --experiment-dir experiments/my_experiment/
```

## 📁 Repository Structure

```
traffic4cast-enhanced/
├── src/
│   ├── data/                 # Data loading and preprocessing
│   │   ├── dataset.py       # Enhanced traffic dataset
│   │   ├── splitter.py      # Data splitting for experiments
│   │   └── spatiotemporal_dataset.py  # Multi-task dataset
│   ├── models/              # Model architectures
│   │   ├── unet.py          # Standard UNet implementation
│   │   └── multitask_unet.py # Multi-task UNet for transfer learning
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Enhanced trainer with TQDM/Wandb
│   │   ├── metrics.py       # Competition metrics (MSE Wiedemann)
│   │   └── callbacks.py     # Checkpointing and logging
│   ├── experiments/         # Experiment runners
│   │   └── spatiotemporal_transfer.py  # Main transfer learning experiment
│   └── utils/               # Utilities and helpers
├── config/                  # Configuration files
│   ├── base.yaml           # Base configuration
│   ├── spatial_config.yaml # Spatial transfer learning
│   └── spatiotemporal_config.yaml  # Enhanced spatio-temporal transfer
├── scripts/                 # Entry points
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Model evaluation
│   └── run_experiments.py  # Batch experiment runner
└── tests/                   # Test suite
```


```

## Acknowledgments

- [Traffic4Cast Competition](https://www.iarai.ac.at/traffic4cast/) for providing the dataset and challenge!
