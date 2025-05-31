import argparse
import logging
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Loss

from custom_data_splitter import CustomDataSplitter, SplitType
# Removed redundant import of TrafficPredictionMetrics to resolve circular import issue
# from evaluation_metrics import TrafficPredictionMetrics
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config
from metrics.mse import mse_loss_wiedemann
from util.monitoring import system_status

import argparse
import binascii
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
import torch_geometric
import tqdm
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from ignite.metrics import RunningAverage
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from baselines.baselines_configs import configs
from baselines.checkpointing import load_torch_model_from_checkpoint
from baselines.checkpointing import save_torch_model_to_checkpoint
from competition.scorecomp import scorecomp
from competition.submission.submission import package_submission
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import GraphTransformer
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import system_status
from util.tar_util import untar_files

def run_model_with_custom_split(
    train_model: torch.nn.Module,
    dataset: Union[T4CDataset, T4CGeometricDataset],
    split_type: SplitType,
    test_city: Optional[str] = None,
    test_year: str = "2020",
    train_year: str = "2019",
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    random_seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    epochs: int = 20,
    dataloader_config: Dict = None,
    optimizer_config: Dict = None,
    device: str = None,
    geometric: bool = False,
    limit: Optional[int] = None,
    experiment_name: str = "t4c_experiment",
    checkpoint_dir: str = "checkpoints",
    logs_dir: str = "logs",
    data_parallel: bool = False,
    device_ids: Optional[list] = None,
    **kwargs,
):
    """
    Run model training and evaluation with custom data splitting.
    
    Parameters
    ----------
    train_model : torch.nn.Module
        Model to train
    dataset : Union[T4CDataset, T4CGeometricDataset]
        Dataset to use
    split_type : SplitType
        Type of split to use (time_based, cross_city, or random)
    test_city : Optional[str]
        City to use for testing in cross_city mode
    test_year : str
        Year to use for testing in time_based mode
    train_year : str
        Year to use for training in time_based mode
    test_fraction : float
        Fraction of data to use for testing
    val_fraction : float
        Fraction of data to use for validation
    random_seed : int
        Random seed for reproducibility
    batch_size : int
        Batch size for data loaders
    num_workers : int
        Number of workers for data loaders
    epochs : int
        Number of epochs to train
    dataloader_config : Dict
        Additional configuration for data loaders
    optimizer_config : Dict
        Configuration for optimizer
    device : str
        Device to use for training
    geometric : bool
        Whether dataset is geometric
    limit : Optional[int]
        Maximum number of samples to use
    experiment_name : str
        Name of experiment for logging
    checkpoint_dir : str
        Directory to save checkpoints
    logs_dir : str
        Directory to save logs
    data_parallel : bool
        Whether to use data parallelism
    device_ids : Optional[list]
        Device IDs for data parallelism
    **kwargs
        Additional arguments
        
    Returns
    -------
    Tuple[torch.nn.Module, str, Dict]
        Trained model, device used, and evaluation metrics
    """
    logging.info(f"Dataset has size {len(dataset)}")
    
    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            logging.info("Using MPS device (Apple Silicon GPU support)")
        elif torch.cuda.is_available():
            device = "cuda"
            logging.info("Using CUDA device")
        else:
            logging.warning("Device not set, using CPU.")
            device = "cpu"
    
    # Setup data parallel if requested
    if torch.cuda.is_available() and data_parallel:
        if torch.cuda.device_count() > 1:
            train_model = torch.nn.DataParallel(train_model, device_ids=device_ids)
            logging.info(f"Using {len(train_model.device_ids)} GPUs: {train_model.device_ids}!")
            device = f"cuda:{train_model.device_ids[0]}"
    
    # Create data splitter and get data loaders
    data_splitter = CustomDataSplitter(
        dataset=dataset,
        split_type=split_type,
        test_city=test_city,
        test_year=test_year,
        train_year=train_year,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        random_seed=random_seed,
        batch_size=batch_size,
        num_workers=num_workers,
        dataloader_config=dataloader_config or {},
        geometric=geometric,
        limit=limit,
    )
    data_loaders = data_splitter.get_data_loaders()
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    # Setup optimizer
    if optimizer_config is None:
        optimizer_config = {}
    if "lr" not in optimizer_config:
        optimizer_config["lr"] = 1e-4
    
    optimizer = optim.Adam(train_model.parameters(), **optimizer_config)
    train_model = train_model.to(device)
    
    # Setup loss function
    loss_fn = F.mse_loss
    
    # Create directories
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"{experiment_name}_{split_type.value}_{experiment_time}"
    
    checkpoint_path = Path(checkpoint_dir) / experiment_path
    logs_path = Path(logs_dir) / experiment_path
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = {
        "experiment_name": experiment_name,
        "split_type": split_type.value,
        "test_city": test_city,
        "test_year": test_year,
        "train_year": train_year,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "random_seed": random_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer_config": optimizer_config,
        "geometric": geometric,
        "device": device,
        "timestamp": experiment_time,
    }
    
    with open(logs_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Training
    if geometric:
        metrics, best_model = train_pure_torch(
            device=device,
            epochs=epochs,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=train_model,
            checkpoint_path=checkpoint_path,
            logs_path=logs_path,
        )
    else:
        metrics = train_ignite(
            device=device,
            epochs=epochs,
            loss=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_model=train_model,
            checkpoint_path=checkpoint_path,
            logs_path=logs_path,
        )
        best_model = train_model  # ignite handles checkpointing internally
    
    logging.info(f"End training of model {train_model} on {device} for {epochs} epochs")
    logging.info(f"Best metrics: {metrics}")
    
    # Save final metrics
    with open(logs_path / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    return best_model, device, metrics

def train_ignite(device, epochs, loss, optimizer, train_loader, val_loader, train_model):
    # Validator
    validation_evaluator = create_supervised_evaluator(train_model, metrics={"val_loss": Loss(loss)}, device=device)
    # Trainer
    trainer = create_supervised_trainer(train_model, optimizer, loss, device=device)
    train_evaluator = create_supervised_evaluator(train_model, metrics={"loss": Loss(loss)}, device=device)
    run_id = binascii.hexlify(os.urandom(15)).decode("utf-8")
    artifacts_path = os.path.join(os.path.curdir, f"artifacts/{run_id}")
    logs_path = os.path.join(artifacts_path, "tensorboard")
    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints")
    RunningAverage(output_transform=lambda x: x).attach(trainer, name="loss")
    pbar = ProgressBar(persist=True, bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]{rate_fmt}")
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_STARTED)  # noqa
    def log_epoch_start(engine: Engine):
        logging.info(f"Started epoch {engine.state.epoch}")
        logging.info(system_status())

    @trainer.on(Events.EPOCH_COMPLETED)  # noqa
    def log_epoch_summary(engine: Engine):
        # Training
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        train_avg_loss = metrics["loss"]

        # Validation
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        val_avg_loss = metrics["val_loss"]

        msg = f"Epoch summary for epoch {engine.state.epoch}: loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}\n"
        pbar.log_message(msg)
        logging.info(msg)
        logging.info(system_status())

    tb_logger = TensorboardLogger(log_dir=logs_path)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(train_model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach_output_handler(
        train_evaluator, event_name=Events.EPOCH_COMPLETED, tag="train", metric_names=["loss"], global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        validation_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["val_loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    to_save = {"train_model": train_model, "optimizer": optimizer}
    checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoints_dir, create_dir=True, require_empty=False), n_saved=1)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    # Run Training
    logging.info("Start training of train_model %s on %s for %s epochs", train_model, device, epochs)
    logging.info(f"tensorboard --logdir={artifacts_path}")
    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()

def train_pure_torch(
    device, 
    epochs, 
    optimizer, 
    train_loader, 
    val_loader, 
    test_loader, 
    model, 
    checkpoint_path, 
    logs_path
):
    """
    Training function for geometric models using pure PyTorch.
    
    Parameters
    ----------
    device : str
        Device to use for training
    epochs : int
        Number of epochs to train
    optimizer : torch.optim.Optimizer
        Optimizer to use
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader
    model : torch.nn.Module
        Model to train
    checkpoint_path : Path
        Path to save checkpoints
    logs_path : Path
        Path to save logs
        
    Returns
    -------
    Tuple[Dict, torch.nn.Module]
        Final metrics and best model
    """
    best_val_loss = float('inf')
    best_model_state = None
    
    # Initialize metrics tracking
    all_metrics = {
        'train_loss': [],
        'val_loss': [],
        'test_metrics': {}
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = _train_epoch_pure_torch(train_loader, device, model, optimizer)
        
        # Validation
        model.eval()
        val_loss = _eval_pure_torch(val_loader, device, model)
        
        # Save metrics
        all_metrics['train_loss'].append(train_loss)
        all_metrics['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            model_path = checkpoint_path / f"model_best_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            
        # Log metrics
        log = f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        logging.info(log)
        
        # Save epoch metrics
        with open(logs_path / f"metrics_epoch_{epoch}.json", "w") as f:
            json.dump({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss)
            }, f, indent=4)
    
    # Test best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_metrics = _test_pure_torch(test_loader, device, model)
    all_metrics['test_metrics'] = test_metrics
    
    # Save final model
    final_model_path = checkpoint_path / "model_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
    }, final_model_path)
    
    logging.info(f"Final test metrics: {test_metrics}")
    
    return test_metrics, model


def _train_epoch_pure_torch(loader, device, model, optimizer):
    """Train for one epoch using PyTorch."""
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, input_data in enumerate(loader):
        if isinstance(input_data, torch.geometric.data.Data):
            input_data = input_data.to(device)
            ground_truth = input_data.y
        else:
            input_data, ground_truth = input_data
            input_data = input_data.to(device)
            ground_truth = ground_truth.to(device)
        
        optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        output = model(input_data)
        loss = criterion(output, ground_truth)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss)
        batch_count += 1
        
        if batch_idx % 50 == 0:
            logging.info(f"Batch {batch_idx}/{len(loader)}, Loss: {float(loss):.4f}")
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def _eval_pure_torch(loader, device, model):
    """Evaluate the model using PyTorch."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for input_data in loader:
            if isinstance(input_data, torch.geometric.data.Data):
                input_data = input_data.to(device)
                ground_truth = input_data.y
            else:
                input_data, ground_truth = input_data
                input_data = input_data.to(device)
                ground_truth = ground_truth.to(device)
            
            criterion = torch.nn.MSELoss()
            output = model(input_data)
            loss = criterion(output, ground_truth)
            
            total_loss += float(loss)
            batch_count += 1
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def _test_pure_torch(loader, device, model):
    """Test the model and compute comprehensive metrics."""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for input_data in loader:
            if isinstance(input_data, torch.geometric.data.Data):
                input_data = input_data.to(device)
                ground_truth = input_data.y
            else:
                input_data, ground_truth = input_data
                input_data = input_data.to(device)
                ground_truth = ground_truth.to(device)
            
            output = model(input_data)
            
            # Move to CPU for numpy conversion
            output = output.cpu()
            ground_truth = ground_truth.cpu()
            
            all_outputs.append(output)
            all_targets.append(ground_truth)
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    metrics = calculate_metrics(all_outputs, all_targets)
    return metrics


def calculate_metrics(predictions, targets, static_mask=None):
    """
    Calculate comprehensive metrics for traffic prediction evaluation.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    targets : np.ndarray
        Ground truth values
    static_mask : np.ndarray, optional
        Static mask for road network (1 for road, 0 otherwise)
        
    Returns
    -------
    Dict
        Dictionary with all computed metrics
    """
    # Convert to torch tensors for more efficient computation
    pred_tensor = torch.from_numpy(predictions).float()
    target_tensor = torch.from_numpy(targets).float()
    
    # Get channel indices
    VOL_CHANNELS = [0, 2, 4, 6]
    SPEED_CHANNELS = [1, 3, 5, 7]
    
    # Overall metrics (standard MSE)
    mse = F.mse_loss(pred_tensor, target_tensor).item()
    rmse = np.sqrt(mse)
    
    # Wiedemann MSE (special treatment for traffic data)
    wiedemann_mse = mse_loss_wiedemann(pred_tensor, target_tensor).item()
    
    # Masked MSE (road pixels only)
    if static_mask is not None:
        # Adjust mask shape if needed
        mask_tensor = torch.from_numpy(static_mask).float()
        if mask_tensor.ndim != pred_tensor.ndim:
            # Broadcast mask to match prediction shape
            mask_shape = list(pred_tensor.shape)
            mask_tensor = mask_tensor.view(*mask_tensor.shape, *([1] * (pred_tensor.ndim - mask_tensor.ndim)))
            mask_tensor = mask_tensor.expand(*mask_shape)
        
        # Apply mask
        pred_masked = pred_tensor * mask_tensor
        target_masked = target_tensor * mask_tensor
        
        # Calculate masked MSE
        mask_elements = torch.sum(mask_tensor).item()
        if mask_elements > 0:
            masked_mse = torch.sum(torch.square(pred_masked - target_masked)).item() / mask_elements
            masked_rmse = np.sqrt(masked_mse)
        else:
            masked_mse = masked_rmse = 0.0
    else:
        masked_mse = masked_rmse = None
    
    # Volume metrics
    volume_pred = pred_tensor[..., VOL_CHANNELS]
    volume_true = target_tensor[..., VOL_CHANNELS]
    volume_mse = F.mse_loss(volume_pred, volume_true).item()
    volume_rmse = np.sqrt(volume_mse)
    
    # Volume accuracy: classified correctly as traffic or no traffic (0 vs >0)
    vol_pred_binary = (volume_pred > 0).float()
    vol_true_binary = (volume_true > 0).float()
    volume_accuracy = torch.mean((vol_pred_binary == vol_true_binary).float()).item()
    
    # Speed metrics (only consider cells with non-zero volume)
    speed_mse_values = []
    speed_correct_total = 0
    speed_total = 0
    speed_tolerance = 10.0  # Tolerance in speed units
    
    for i, channel in enumerate(SPEED_CHANNELS):
        speed_pred = pred_tensor[..., channel]
        speed_true = target_tensor[..., channel]
        volume_true = target_tensor[..., VOL_CHANNELS[i]]
        
        # Create mask for non-zero volume
        mask = (volume_true > 0)
        mask_sum = torch.sum(mask).item()
        if mask_sum == 0:
            continue
            
        # Apply mask
        speed_pred_masked = speed_pred[mask]
        speed_true_masked = speed_true[mask]
        
        # Calculate MSE
        channel_mse = torch.mean(torch.square(speed_pred_masked - speed_true_masked)).item()
        speed_mse_values.append(channel_mse)
        
        # Speed accuracy: within tolerance
        speed_correct = torch.sum((torch.abs(speed_pred_masked - speed_true_masked) <= speed_tolerance).float()).item()
        speed_correct_total += speed_correct
        speed_total += mask_sum
    
    if speed_total > 0 and speed_mse_values:
        speed_mse = sum(speed_mse_values) / len(speed_mse_values)
        speed_rmse = np.sqrt(speed_mse)
        speed_accuracy = speed_correct_total / speed_total
    else:
        speed_mse = speed_rmse = speed_accuracy = 0.0
    
    # Build metrics dictionary
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "wiedemann_mse": float(wiedemann_mse),
        "masked_mse": float(masked_mse) if masked_mse is not None else None,
        "masked_rmse": float(masked_rmse) if masked_rmse is not None else None,
        "volume_mse": float(volume_mse),
        "volume_rmse": float(volume_rmse),
        "volume_accuracy": float(volume_accuracy),
        "speed_mse": float(speed_mse),
        "speed_rmse": float(speed_rmse),
        "speed_accuracy": float(speed_accuracy),
    }
    return metrics