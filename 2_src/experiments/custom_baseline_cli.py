import argparse
import logging
import sys
from pathlib import Path

from data.splitter import SplitType
from baselines.custom_training import run_model_with_custom_split
from baselines.baselines_configs import configs
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config


def create_parser():
    parser = argparse.ArgumentParser(description="Custom training script for Traffic4Cast with advanced split options")

    # Model arguments
    parser.add_argument("--model_str", type=str, help="One of configurations, e.g. 'unet'.", default="unet")
    parser.add_argument("--resume_checkpoint", type=str, help="Torch pt file to be re-loaded.", default=None)
    
    # Data arguments
    parser.add_argument("--data_raw_path", type=str, help="Base dir of raw data", default="./data/raw")
    parser.add_argument("--file_filter", type=str, default=None, help="Filter files in the dataset. Defaults to '**/*8ch.h5'")
    
    # Split type arguments
    parser.add_argument("--split_type", type=str, choices=["time_based", "cross_city", "random"], default="time_based", help="Type of split to use")
    parser.add_argument("--test_city", type=str, default=None, help="City to use for testing in cross_city mode")
    parser.add_argument("--test_year", type=str, default="2020", help="Year to use for testing in time_based mode")
    parser.add_argument("--train_year", type=str, default="2019", help="Year to use for training in time_based mode")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of data to use for validation")
    
    # Training arguments
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset size at this limit")
    parser.add_argument("--device", type=str, default=None, help="Force usage of device")
    parser.add_argument("--device_ids", nargs="*", default=None, help="Whitelist of device ids")
    parser.add_argument("--data_parallel", action="store_true", help="Use DataParallel for multi-GPU training")
    
    # Loss function arguments
    parser.add_argument("--use_wiedemann_loss", action="store_true", 
                        help="Use MSELossWiedemann for traffic-specific training")
    
    # Mask arguments
    parser.add_argument("--use_static_mask", action="store_true",
                        help="Use static road mask for evaluation metrics")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="t4c_experiment", help="Name of experiment for logging")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory to save logs")
    
    # Logging arguments
    parser.add_argument("-log", "--loglevel", default="info", help="Provide logging level. Example --loglevel debug, default=info")
    
    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Configure logging
    t4c_apply_basic_logging_config(loglevel=args.loglevel.upper())
    
    # Get model configuration
    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint
    
    # Get split type
    split_type = SplitType(args.split_type)
    
    # Setup dataset
    dataset_config = configs[model_str].get("dataset_config", {})
    geometric = configs[model_str].get("geometric", False)
    
    logging.info(f"Loading dataset from {args.data_raw_path}")
    if geometric:
        dataset = T4CGeometricDataset(
            root=str(Path(args.data_raw_path).parent),
            file_filter=args.file_filter,
            num_workers=args.num_workers,
            **dataset_config
        )
    else:
        dataset = T4CDataset(
            root_dir=args.data_raw_path,
            file_filter=args.file_filter,
            **dataset_config
        )
    
    logging.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Load static masks for evaluation if requested or in cross-city mode
    static_mask = None
    if args.use_static_mask or (split_type == SplitType.CROSS_CITY and args.test_city):
        from custom_training import load_static_mask
        city = args.test_city if args.test_city else "BARCELONA"  # Default if not specified
        static_mask = load_static_mask(city, args.data_raw_path)
        logging.info(f"Loaded static mask for {city}")
    
    # Create model
    logging.info(f"Creating model: {model_str}")
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config)
    
    # Load checkpoint if specified
    if resume_checkpoint:
        logging.info(f"Loading checkpoint: {resume_checkpoint}")
        from baselines.checkpointing import load_torch_model_from_checkpoint
        model = load_torch_model_from_checkpoint(resume_checkpoint, model)
    
    # Get dataloader and optimizer configs
    dataloader_config = configs[model_str].get("dataloader_config", {})
    optimizer_config = configs[model_str].get("optimizer_config", {})
    
    # Setup custom loss function if requested
    loss_function = None
    if args.use_wiedemann_loss:
        from evaluation_metrics import MSELossWiedemann
        loss_function = MSELossWiedemann()
        logging.info("Using Wiedemann MSE loss function")
    
    # Run training with custom split
    logging.info(f"Starting training with {split_type.value} split")
    run_model_with_custom_split(
        train_model=model,
        dataset=dataset,
        split_type=split_type,
        test_city=args.test_city,
        test_year=args.test_year,
        train_year=args.train_year,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        dataloader_config=dataloader_config,
        optimizer_config=optimizer_config,
        device=args.device,
        geometric=geometric,
        limit=args.limit,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir,
        data_parallel=args.data_parallel,
        device_ids=args.device_ids,
        static_mask=static_mask,
        loss_function=loss_function,
    )


if __name__ == "__main__":
    main()