import os
import subprocess
import sys

import numpy as np
from ase import units
from ase.io import write
from ase.io.trajectory import Trajectory

from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES

FES_SCRIPT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "third_party", "FES_from_State.py")


def read_plumed_table(path):
    """Column names from the first "#! FIELDS" line and the numeric rows of a PLUMED output file (COLVAR, fes.dat).

    Returns (column names, (n_rows, n_columns) array); raises if the column count does not match FIELDS.
    """
    with open(path) as input_file:
        column_names = next(
            line.split()[2:] for line in input_file if line.startswith("#! FIELDS"))
    table_values = np.loadtxt(path, comments="#", ndmin=2)
    if table_values.shape[1] != len(column_names):
        raise ValueError(
            f"{path}: {table_values.shape[1]} columns but FIELDS lists {len(column_names)}")
    return column_names, table_values


def run(settings, output_dir, inputs):
    """Frame of an OPES trajectory closest in CV space to the minimum of the FES reconstructed from STATE.

    STATE and COLVAR are read from the directory of the trajectory; colvar_stride = COLVAR rows per trajectory frame.
    Returns [<output_dir>/<name>.traj]; also writes <name>_fes.dat.
    """
    if len(inputs) != 1:
        raise ValueError(
            f"opes_minimum needs exactly one trajectory, got {inputs}")
    merged_settings, output_prefix = write_merged_settings(
        STAGES["opes_minimum"], settings, output_dir, inputs)
    opes_run_dir = os.path.dirname(inputs[0])

    fes_file = f"{output_prefix}_fes.dat"
    # FES_from_State.py takes kT in the energy unit of the STATE file, eV here
    fes_command = [sys.executable, FES_SCRIPT, "--state", os.path.join(opes_run_dir, merged_settings["state_file"]), "--outfile", fes_file,
                   "--kt", str(merged_settings["temperature_K"] * units.kB)]
    if merged_settings["fes_bins"] is not None:
        fes_command += ["--bin", str(merged_settings["fes_bins"])]
    subprocess.run(fes_command, check=True)

    fes_column_names, fes_table = read_plumed_table(fes_file)
    # the last FES column is the free energy, the others are the CVs
    cv_names = fes_column_names[:-1]
    fes_minimum_cv_values = fes_table[np.argmin(fes_table[:, -1]), :-1]

    colvar_column_names, colvar_table = read_plumed_table(
        os.path.join(opes_run_dir, merged_settings["colvar_file"]))
    cvs_missing_in_colvar = set(cv_names) - set(colvar_column_names)
    if cvs_missing_in_colvar:
        raise ValueError(f"CVs {sorted(cvs_missing_in_colvar)} of the FES are not in {merged_settings['colvar_file']} "
                         f"fields {colvar_column_names}")
    cv_values_per_frame = colvar_table[::merged_settings["colvar_stride"], [
        colvar_column_names.index(cv_name) for cv_name in cv_names]]

    opes_trajectory = Trajectory(inputs[0])
    if len(cv_values_per_frame) != len(opes_trajectory):
        raise ValueError(f"{len(colvar_table)} COLVAR rows with colvar_stride {merged_settings['colvar_stride']} give "
                         f"{len(cv_values_per_frame)} frames, but {inputs[0]} has {len(opes_trajectory)}")
    cv_distance_to_minimum = np.linalg.norm(
        cv_values_per_frame - fes_minimum_cv_values, axis=1)
    closest_frame_index = int(np.argmin(cv_distance_to_minimum))
    closest_frame = opes_trajectory[closest_frame_index]
    opes_trajectory.close()

    write(f"{output_prefix}.traj", closest_frame)
    print(f"opes_minimum: FES minimum {dict(zip(cv_names, fes_minimum_cv_values.tolist()))}, frame {closest_frame_index} with "
          f"{dict(zip(cv_names, cv_values_per_frame[closest_frame_index].tolist()))} "
          f"(distance {cv_distance_to_minimum[closest_frame_index]:.4g}) -> {output_prefix}.traj", flush=True)
    outputs = [f"{output_prefix}.traj"]
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "opes_minimum", sys.argv[2:])
    raise_on_problems("opes_minimum", settings, inputs)
    run(settings, ".", inputs)
