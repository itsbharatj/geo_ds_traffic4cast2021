# Custom Training Pipeline for Traffic4Cast

This custom training pipeline provides advanced data splitting strategies and evaluation metrics for the Traffic4Cast challenge. It allows you to perform time-based splitting (e.g., train on 2019 data, test on 2020 data) and cross-city splitting (e.g., train on data from some cities, test on a different city), which are particularly useful for domain adaptation tasks.

## Features

- **Custom Data Splitting**:
  - **Time-based**: Train on one time period, test on another (e.g., pre-COVID vs. COVID periods)
  - **Cross-city**: Train on data from some cities, test on others for spatial generalization
  - **Combined**: Mix time-based and cross-city approaches for real-world generalization

- **Traffic-Specific Metrics**:
  - **Road-only Evaluation**: Apply static road masks to focus evaluation on relevant areas
  - **Wiedemann MSE Loss**: Special loss function for traffic data that better handles zero-volume cases
  - **Volume/Speed Separate Metrics**: Separate metrics for traffic volume and speed channels
  - **Volume Accuracy**: Evaluate correctness of traffic/no-traffic classification
  - **Speed Accuracy**: Evaluate accuracy of speed predictions within tolerance

- **Training Infrastructure**:
  - Support for both PyTorch Ignite and pure PyTorch training loops
  - TensorBoard logging and visualization
  - Advanced checkpointing for best model saving
  - Configurable via command line or programmatically

## Installation

1. Set up the base Traffic4Cast environment according to the original instructions
2. Copy these custom training files to your project directory:
   - `custom_data_splitter.py`
   - `custom_training.py`
   - `custom_training_cli.py`
   - `evaluation_metrics.py`

## Usage Examples

### Command Line Interface

```bash
# Time-based split (train on 2019, test on 2020)
python custom_training_cli.py \
  --model_str unet \
  --data_raw_path ./data/raw \
  --split_type time_based \
  --train_year 2019 \
  --test_year 2020 \
  --batch_size 4 \
  --epochs 20 \
  --experiment_name time_based_experiment

# Cross-city split (train on all cities except Barcelona, test on Barcelona)
python custom_training_cli.py \
  --model_str unet \
  --data_raw_path ./data/raw \
  --split_type cross_city \
  --test_city BARCELONA \
  --batch_size 4 \
  --epochs 20 \
  --use_static_mask \
  --experiment_name cross_city_experiment

# Using Wiedemann loss for traffic-specific training
python custom_training_cli.py \
  --model_str unet \
  --data_raw_path ./data/raw \
  --split_type time_based \
  --use_wiedemann_loss \
  --batch_size 4 \
  --epochs 20 \
  --experiment_name wiedemann_loss_experiment
```

### Programmatic Usage

```python
from custom_data_splitter import SplitType
from custom_training import run_model_with_custom_split
from baselines.baselines_configs import configs
from data.dataset.dataset import T4CDataset

# Load model and dataset
model_str = "unet"
model_class = configs[model_str]["model_class"]
model_config = configs[model_str].get("model_config", {})
model = model_class(**model_config)

dataset = T4CDataset(
    root_dir="./data/raw",
    file_filter="**/*8ch.h5",
    **configs[model_str].get("dataset_config", {})
)

# Run training with time-based split
run_model_with_custom_split(
    train_model=model,
    dataset=dataset,
    split_type=SplitType.TIME_BASED,
    test_year="2020",
    train_year="2019",
    batch_size=4,
    epochs=20,
    experiment_name="my_experiment"
)
```

## Metrics Explained

### Standard Metrics
- **MSE**: Mean Squared Error across all pixels and channels
- **RMSE**: Root Mean Squared Error

### Traffic-Specific Metrics
- **Wiedemann MSE**: MSE with special handling for zero-volume cases
- **Masked MSE/RMSE**: Error metrics considering only road pixels (using static mask)
- **Volume MSE/RMSE**: Error metrics for volume channels only
- **Speed MSE/RMSE**: Error metrics for speed channels only (where volume > 0)
- **Volume Accuracy**: Percentage of pixels correctly classified as having traffic or not
- **Speed Accuracy**: Percentage of pixels with speed predictions within tolerance

## Custom Data Splitting Details

### Time-based Split
This approach splits the data based on the year/time period:
- **Train**: Data from one time period (e.g., 2019)
- **Validation**: A subset of the train period data
- **Test**: Data from another time period (e.g., 2020)

This is useful for temporal domain adaptation, such as adapting from pre-COVID traffic patterns to COVID-era patterns.

### Cross-city Split
This approach splits the data based on cities:
- **Train**: Data from selected cities (e.g., all cities except Barcelona)  
- **Validation**: A subset of the train cities' data
- **Test**: Data from the held-out city (e.g., Barcelona)

This is useful for spatial domain adaptation, testing how well models generalize to new geographic regions.

### Combined Approach
For the most challenging generalization scenario:
- **Train**: Data from pre-COVID (2019) in some cities
- **Test**: Data from COVID period (2020) in different cities

This tests both temporal and spatial generalization capabilities.

## Extending the Pipeline

The modular design makes it easy to extend with:
- New data splitting strategies by subclassing `CustomDataSplitter`
- Custom loss functions by implementing PyTorch loss modules
- Additional metrics by extending `TrafficPredictionMetrics`

## Credits

Our pipeline builds upon the original Traffic4Cast competition codebase and adds custom functionality for domain adaptation experiments.