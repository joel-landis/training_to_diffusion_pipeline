import hashlib
import json
import os

import numpy as np


def cache_key(**key_fields):
    """16-character hash of the JSON of key_fields; identical fields give the identical key."""
    return hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]


def trajectory_cache_fields(trajectory_path):
    """Path, size and modification time of a trajectory as cache_key fields; a trajectory overwritten under the same path gets a new key."""
    trajectory_stat = os.stat(trajectory_path)
    return {"traj": trajectory_path, "size": trajectory_stat.st_size, "mtime": trajectory_stat.st_mtime_ns}


def load_or_compute_cached(path, compute_arrays):
    """Dict of arrays from the npz at path if it exists, otherwise compute_arrays() saved to path."""
    if os.path.exists(path):
        print(f"using cache {path}", flush=True)
        with np.load(path) as cached_arrays:
            return {key: cached_arrays[key] for key in cached_arrays.files}
    computed_arrays = compute_arrays()
    # write and rename, so an interruption never leaves a truncated cache file
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as output_file:
        np.savez(output_file, **computed_arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)
    return computed_arrays


def frames_from_ps(time_ps, dt_fs):
    """Number of frames closest to time_ps for a frame spacing of dt_fs."""
    return round(time_ps * 1000.0 / dt_fs)
