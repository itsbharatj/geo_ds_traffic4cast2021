# src/utils/reproducibility.py
import random
import numpy as np
import torch
import os

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed}")
