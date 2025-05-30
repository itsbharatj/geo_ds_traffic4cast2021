import logging
from pathlib import Path
from typing import Optional
from typing import Union

import h5py
import numpy as np
import torch

# -----------------------------------------------------------------
# h5 file support
# -----------------------------------------------------------------

def load_h5_file(file_path: Union[str, Path], sl: Optional[slice] = None, to_torch: bool = False) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.

    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl])
        else:
            data = np.array(data)
        if to_torch:
            data = torch.from_numpy(data)
            data = data.to(dtype=torch.float)
        return data


def write_data_to_h5(data: np.ndarray, filename: Union[str, Path], compression="gzip", compression_level=9, dtype="uint8", verbose=False):
    """write data in gzipped h5 format.

    Parameters
    ----------
    data
    filename
    compression
    compression_level
    verbose
    """
    with h5py.File(filename if isinstance(filename, str) else str(filename), "w", libver="latest") as f:
        if data.dtype != dtype:
            logging.warning(f"Found data with {data.dtype}, expected {dtype}.")
        if verbose:
            print(f"writing {filename} ...")
        f.create_dataset(
            # `chunks=(1, *data.shape[1:])`: optimize for row access!
            "array",
            shape=data.shape,
            data=data,
            chunks=(1, *data.shape[1:]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_level,
        )
        if verbose:
            print(f"... done writing {filename}")
