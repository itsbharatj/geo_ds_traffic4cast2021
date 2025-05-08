import h5py
import numpy as np
from pathlib import Path
from typing import Union


def explore_h5_file(file_path: Union[str, Path]):
    """Explore the structure of an H5 file and print details about its datasets."""
    file_path = str(file_path)  # Ensure file_path is a string
    try:
        with h5py.File(file_path, "r") as h5_file:
            print(f"Exploring H5 file: {file_path}")
            print("-" * 50)

            def explore_group(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print(f"{prefix}Group: {key}")
                        explore_group(item, prefix + "  ")
                    elif isinstance(item, h5py.Dataset):
                        print(f"{prefix}Dataset: {key}")
                        print(f"{prefix}  Shape: {item.shape}")
                        print(f"{prefix}  Dtype: {item.dtype}")
                        # Print the first 100 lines if the dataset is small enough
                        if item.ndim == 1 or item.ndim == 2:
                            data = item[:100] if item.size > 100 else item[:]
                            print(f"{prefix}  First 100 lines of data:")
                            print(data)
                        print("-" * 50)

            explore_group(h5_file)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python explore_h5.py <h5_file1> [<h5_file2> ...]")
        sys.exit(1)

    for h5_file in sys.argv[1:]:
        explore_h5_file(h5_file)