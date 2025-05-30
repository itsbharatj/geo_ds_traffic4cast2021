# scripts/run_experiments.py
"""
Run multiple experiments in sequence

Usage:
    python scripts/run_experiments.py
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(config_path, experiment_name):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", config_path,
        "--experiment-name", experiment_name
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {experiment_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed with return code {e.returncode}")
        return False

def main():
    experiments = [
        ("config/debug.yaml", "debug_test"),
        ("config/spatial_transfer.yaml", "spatial_transfer_main"),
        ("config/spatiotemporal_transfer.yaml", "spatiotemporal_transfer_main"),
    ]
    
    results = {}
    
    for config_path, experiment_name in experiments:
        success = run_experiment(config_path, experiment_name)
        results[experiment_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for experiment_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{experiment_name}: {status}")
    
    total_success = sum(results.values())
    print(f"\nTotal: {total_success}/{len(results)} experiments successful")

if __name__ == "__main__":
    main()