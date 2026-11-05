"""Entry functions of the pipeline stages, one per operation (the TOML stage's task value).

Each takes the work unit's output directory, its input paths and the stage settings as keyword arguments, and returns the
output paths. Building blocks shared by several stages live in the backend modules.
"""
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import units
from ase.build import sort
from ase.io import read, write
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp
from torch_geometric.nn import fps

from pipeline import vasp_jobs
from pipeline.analyze_coordination import coordination_analysis
from pipeline.analyze_diffusion import diffusion_analysis
from pipeline.io_utils import unique_split_names, write_csv
from pipeline.ase_simulation import build_dispersed_cluster, relax_with_bfgs, run_bussi_md
from pipeline.plot_diffusion import (load_coordination_means, load_diffusion_results, plot_coordination_trends,
                                     plot_diffusion_ratios, plot_highlights, plot_overviews, write_coordination_correlations)
from pipeline.plot_structure import (plot_com_distance_densities, plot_com_distance_traces, plot_metal_atoms_per_c,
                                     plot_pair_distance_peaks)
from pipeline.plot_utils import PLOT_STYLE, coordination_files_by_system

FES_SCRIPT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "third_party", "FES_from_State.py")


def generate_clusters(output_dir, inputs, *, name, composition, radius, temperature_K, timestep_fs, taut_fs,
                     equilibration_nsteps, write_interval_fs, log_interval_fs, c_c_separation=3.5, cov_scale=0.8,
                     max_attempts=50000, vacuum=10.0, seed=None, fmax=0.1, opt_max_steps=200,
                     remove_rigid_body_interval_fs=10.0, **mace_settings):
    """One cluster as a starting point for MD: random placement, BFGS relaxation, Bussi equilibration.

    Returns [<output_dir>/<name>.traj], centred with vacuum Å on each side. inputs is not used.
    """
    output_prefix = os.path.join(output_dir, name)
    mace_calculator = mace_mp(**mace_settings)
    print(
        f"cluster: composition {composition}, radius {radius} A, seed {seed}", flush=True)
    print(f"cluster: calculator {mace_settings}", flush=True)
    random_number_generator = np.random.default_rng(seed)
    shuffled_symbols = [element for element,
                        count in composition.items() for _ in range(count)]
    random_number_generator.shuffle(shuffled_symbols)

    cluster = build_dispersed_cluster(shuffled_symbols, radius, c_c_separation, cov_scale, max_attempts,
                                      random_number_generator)
    cluster.center(vacuum=vacuum)
    write(f"{output_prefix}_pre_opt.xyz", cluster, format="xyz")

    bfgs_converged, bfgs_steps, final_max_force = relax_with_bfgs(
        cluster, mace_calculator, fmax, opt_max_steps, output_prefix, f"{name} BFGS")
    print(f"BFGS converged to fmax={fmax}: {bfgs_converged} ({bfgs_steps} steps, "
          f"final fmax {final_max_force:.3f} eV/A)", flush=True)
    sorted_cluster = sort(cluster, tags=cluster.numbers)
    # BFGS shrinks the cluster; re-centre to keep the vacuum
    sorted_cluster.center(vacuum=vacuum)
    write(f"{output_prefix}_relaxed.traj", sorted_cluster)

    run_bussi_md(sorted_cluster, mace_calculator, temperature_K=temperature_K, timestep_fs=timestep_fs, taut_fs=taut_fs,
                 nsteps=equilibration_nsteps, write_interval_fs=write_interval_fs, log_interval_fs=log_interval_fs,
                 remove_rigid_body_interval_fs=remove_rigid_body_interval_fs,
                 random_number_generator=random_number_generator, output_prefix=f"{output_prefix}_equilibration",
                 progress_label=f"{name} equilibration")
    # back to the box centre after the MD
    sorted_cluster.center(vacuum=vacuum)
    write(f"{output_prefix}.traj", sorted_cluster)
    return [f"{output_prefix}.traj"]


def md(output_dir, inputs, *, name, temperature_K, timestep_fs, taut_fs, nsteps, write_interval_fs, log_interval_fs,
           seed=None, plumed=None, start_structure_format=None, remove_rigid_body_interval_fs=10.0, **mace_settings):
    """Bussi MD from one start structure, optionally biased with a PLUMED template; returns [<output_dir>/<name>.traj].

    The start structure's velocities are not used; it is linked as start<ext> in output_dir. PLUMED output files (COLVAR,
    STATE, ...) are written to output_dir.
    """
    if len(inputs) != 1:
        raise ValueError(
            f"md needs exactly one start structure, got {inputs}")
    output_prefix = os.path.join(output_dir, name)
    start_link = os.path.join(
        output_dir, "start" + os.path.splitext(inputs[0])[1])
    # a resumed work unit keeps the link of its first attempt
    if not os.path.lexists(start_link):
        os.symlink(os.path.relpath(inputs[0], output_dir), start_link)
    start_atoms = read(inputs[0], format=start_structure_format)
    print(f"md: start {inputs[0]} ({len(start_atoms)} atoms, {start_atoms.get_chemical_formula()}), seed {seed}", flush=True)
    print(f"md: calculator {mace_settings}, plumed {plumed}", flush=True)
    run_bussi_md(start_atoms, mace_mp(**mace_settings), temperature_K=temperature_K, timestep_fs=timestep_fs, taut_fs=taut_fs,
                 nsteps=nsteps, write_interval_fs=write_interval_fs, log_interval_fs=log_interval_fs,
                 remove_rigid_body_interval_fs=remove_rigid_body_interval_fs,
                 random_number_generator=np.random.default_rng(seed), output_prefix=output_prefix,
                 progress_label=name, plumed_template=plumed)
    print(f"md: wrote {output_prefix}.traj, {output_prefix}.log", flush=True)
    return [f"{output_prefix}.traj"]


def relax_mlip(output_dir, inputs, *, name, fmax=0.1, opt_max_steps=500, **mace_settings):
    """Relax a copy of every frame whose largest MACE force exceeds fmax, and append the copies to their split.

    Returns one <name>_<split>.traj per input: the original frames, then the relaxed copies marked info["relaxed"] = "mlip".
    Also writes <name>_relax.csv and the BFGS files in <output_dir>/<split>/.
    """
    output_prefix = os.path.join(output_dir, name)
    mace_calculator = mace_mp(**mace_settings)
    relax_rows, outputs = [], []
    for split_path, split in zip(inputs, unique_split_names(inputs)):
        input_frames = read(split_path, index=":")
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        relaxed_copies = []
        for frame_index, frame in enumerate(input_frames):
            frame_copy = frame.copy()
            frame_copy.calc = mace_calculator
            initial_max_force = np.linalg.norm(
                frame_copy.get_forces(), axis=1).max()
            if initial_max_force <= fmax:
                relax_rows.append(
                    [split, frame_index, initial_max_force, "", "", ""])
                continue
            bfgs_converged, bfgs_steps, final_max_force = relax_with_bfgs(
                frame_copy, mace_calculator, fmax, opt_max_steps,
                os.path.join(output_dir, split, f"{frame_index:05d}"), f"{name} {split} {frame_index}")
            # without a calculator, write() stores no MACE energy and forces for the relaxed copy
            frame_copy.calc = None
            frame_copy.info["relaxed"] = "mlip"
            relaxed_copies.append(frame_copy)
            relax_rows.append([split, frame_index, initial_max_force,
                              bfgs_converged, bfgs_steps, final_max_force])
        write(f"{output_prefix}_{split}.traj", input_frames + relaxed_copies)
        outputs.append(f"{output_prefix}_{split}.traj")
        print(f"{name}: {split} {len(input_frames)} frames + {len(relaxed_copies)} relaxed copies "
              f"-> {output_prefix}_{split}.traj", flush=True)
    write_csv(f"{output_prefix}_relax.csv",
              ["split", "frame_index", "initial_fmax", "converged", "steps", "final_fmax"], relax_rows)
    return outputs


def pick_frames_by_fps(atom_descriptors_per_frame, n_frames, atoms_per_frame_factor, device, excluded_frame_indices):
    """Pick up to n_frames frames by farthest point sampling over the descriptors of all atoms of all frames.

    FPS samples atoms_per_frame_factor * n_frames atoms, since several can fall in one frame; excluded_frame_indices are never picked.
    Returns {frame index: index of the atom whose pick selected the frame}, in FPS order.
    """
    # FPS runs over atoms; row r of the stacked descriptors is atom atom_index_of_row[r] of frame frame_index_of_row[r]
    stacked_atom_descriptors = np.vstack(atom_descriptors_per_frame)
    frame_index_of_row = np.repeat(np.arange(len(atom_descriptors_per_frame)), [
                                   len(descriptors) for descriptors in atom_descriptors_per_frame])
    atom_index_of_row = np.concatenate(
        [np.arange(len(descriptors)) for descriptors in atom_descriptors_per_frame])
    n_atoms_to_sample = min(atoms_per_frame_factor *
                            n_frames, len(stacked_atom_descriptors))
    descriptor_tensor = torch.tensor(stacked_atom_descriptors, device=device)
    picked_atom_by_frame = {}
    # fixed start, so the same descriptors always give the same selection
    for fps_row in fps(descriptor_tensor, ratio=n_atoms_to_sample / len(stacked_atom_descriptors), random_start=False).cpu().numpy():
        frame_index = int(frame_index_of_row[fps_row])
        # a frame counts once, at the first of its atoms that FPS picks
        if frame_index in excluded_frame_indices or frame_index in picked_atom_by_frame:
            continue
        picked_atom_by_frame[frame_index] = int(atom_index_of_row[fps_row])
        if len(picked_atom_by_frame) == n_frames:
            break
    return picked_atom_by_frame


def select_frames(output_dir, inputs, *, name, frames_per_traj, device, thin_to=3000, atoms_per_frame_factor=5,
                  val_fraction=0.2, seed=42, exclude_provenance=None, **mace_settings):
    """Select frames_per_traj frames per input trajectory by FPS on MACE descriptors and split them into train and val.

    Frames listed in exclude_provenance, the provenance CSV of an earlier selection, are not picked again.
    Returns [<name>_train.traj, <name>_val.traj]; also writes <name>_provenance.csv.
    """
    output_prefix = os.path.join(output_dir, name)
    mace_calculator = mace_mp(device=device, **mace_settings)
    excluded_trajectory_frames = set()
    if exclude_provenance is not None:
        previous_provenance = pd.read_csv(exclude_provenance)
        excluded_trajectory_frames = set(
            zip(previous_provenance["traj_file"], previous_provenance["orig_index"]))

    selected_frames, provenance_records = [], []
    for trajectory_path in inputs:
        # every len // thin_to-th frame, at most thin_to frames
        trajectory = Trajectory(trajectory_path)
        thinning_stride = max(1, len(trajectory) // thin_to)
        thinned_indices = list(
            range(0, len(trajectory), thinning_stride))[:thin_to]
        thinned_frames = [trajectory[frame_index]
                          for frame_index in thinned_indices]
        trajectory.close()

        energies_per_atom, max_forces, atom_descriptors_per_frame = [], [], []
        for frame in thinned_frames:
            # copy() drops the calculator with the stored MD results, so the stage's model is evaluated
            frame_with_calculator = frame.copy()
            frame_with_calculator.calc = mace_calculator
            energies_per_atom.append(
                frame_with_calculator.get_potential_energy() / len(frame_with_calculator))
            max_forces.append(np.linalg.norm(
                frame_with_calculator.get_forces(), axis=1).max())
            atom_descriptors_per_frame.append(
                mace_calculator.get_descriptors(frame).astype(np.float32))

        excluded_frame_indices = {frame_index for frame_index, original_index in enumerate(thinned_indices)
                                  if (trajectory_path, original_index) in excluded_trajectory_frames}
        # FPS on the model's own atom descriptors spreads the selection over the environments this model distinguishes
        picked_atom_by_frame = pick_frames_by_fps(atom_descriptors_per_frame, frames_per_traj, atoms_per_frame_factor,
                                                  device, excluded_frame_indices)
        if len(picked_atom_by_frame) < frames_per_traj:
            print(
                f"{trajectory_path}: only {len(picked_atom_by_frame)} of {frames_per_traj} frames selected", flush=True)
        for frame_index, atom_index in picked_atom_by_frame.items():
            selected_frames.append(thinned_frames[frame_index])
            provenance_records.append({"traj_file": trajectory_path, "orig_index": thinned_indices[frame_index],
                                       "atom_index": atom_index, "energy_per_atom": energies_per_atom[frame_index],
                                       "max_force": max_forces[frame_index]})

    shuffled_order = np.random.default_rng(
        seed).permutation(len(selected_frames))
    n_val = int(len(shuffled_order) * val_fraction)
    val_positions, train_positions = shuffled_order[:n_val], shuffled_order[n_val:]
    for position in val_positions:
        provenance_records[position]["split"] = "val"
    for position in train_positions:
        provenance_records[position]["split"] = "train"

    write(f"{output_prefix}_train.traj", [
          selected_frames[position] for position in train_positions])
    write(f"{output_prefix}_val.traj", [
          selected_frames[position] for position in val_positions])
    with open(f"{output_prefix}_provenance.csv", "w") as output_file:
        pd.DataFrame(provenance_records).to_csv(output_file, index=False)
        output_file.flush()
        os.fsync(output_file.fileno())
    print(
        f"selected {len(selected_frames)} frames: train {len(train_positions)}, val {len(val_positions)}", flush=True)
    return [f"{output_prefix}_train.traj", f"{output_prefix}_val.traj"]


def finetune(output_dir, inputs, *, name, device, mace_args):
    """Fine-tune MACE with python -m mace.cli.run_train in output_dir; returns [<output_dir>/<mace_args.name>.model].

    inputs: exactly one *_train and one *_val extxyz file. mace_args must contain "name"; train_file, valid_file and device
    are set here.
    """
    reserved_keys_given = {"train_file", "valid_file", "device"} & set(mace_args)
    if reserved_keys_given:
        raise ValueError(
            f"mace_args {sorted(reserved_keys_given)} are set by the stage; use inputs and device")
    split_names = unique_split_names(inputs)
    if sorted(split_names) != ["train", "val"]:
        raise ValueError(
            f"inputs {inputs} must be exactly one _train and one _val file")
    train_file, valid_file = (os.path.abspath(
        inputs[split_names.index(split)]) for split in ("train", "val"))

    # true -> bare --key, false -> omitted, anything else -> --key str(value). MACE options parsed with
    # str2bool (e.g. multiheads_finetuning) need the string "True"/"False"; a TOML boolean is only right for store_true
    # options (e.g. restart_latest).
    run_train_command = [sys.executable, "-m", "mace.cli.run_train", "--train_file", train_file,
                         "--valid_file", valid_file, "--device", device]
    for key, value in mace_args.items():
        if value is True:
            run_train_command.append(f"--{key}")
        elif value is not False:
            run_train_command += [f"--{key}", str(value)]
    print(f"{name}: {' '.join(run_train_command)}", flush=True)

    subprocess.run(run_train_command, cwd=output_dir, check=True)
    model_path = os.path.join(output_dir, f"{mace_args['name']}.model")
    if not os.path.exists(model_path):
        raise RuntimeError(f"run_train finished without {model_path}")
    return [model_path]


def relax_vasp(output_dir, inputs, *, name, **vasp_settings):
    """Relax frames with VASP, one SLURM job per frame; requires ionic convergence.

    Per split, frames marked info["relaxed"] (e.g. by relax_mlip) are replaced by their VASP-relaxed structure; if none
    are marked, relaxed copies of all frames are appended. Returns one <name>_<split>.traj per input.
    """
    output_prefix = os.path.join(output_dir, name)
    split_names = unique_split_names(inputs)
    frames_by_split, frames_to_relax = {}, []
    for split_path, split in zip(inputs, split_names):
        frames = read(split_path, index=":")
        marked_indices = [frame_index for frame_index,
                          frame in enumerate(frames) if "relaxed" in frame.info]
        if not marked_indices:
            marked_indices = list(range(len(frames), 2 * len(frames)))
            frames += [frame.copy() for frame in frames]
        frames_by_split[split] = frames
        frames_to_relax += [(split, frame_index, frames[frame_index])
                            for frame_index in marked_indices]
    frame_dirs = vasp_jobs.run_vasp_frames({"name": name, **vasp_settings}, output_dir, frames_to_relax,
                                           vasp_jobs.RELAX_CONVERGED_MARKER, continue_from_contcar=True)
    for (split, frame_index, _), frame_dir in zip(frames_to_relax, frame_dirs):
        relaxed_atoms = read(os.path.join(
            frame_dir, "OUTCAR"), format="vasp-out", index=-1)
        # relaxation energies are not kept; labels come from label_vasp
        relaxed_atoms.calc = None
        relaxed_atoms.info["relaxed"] = "vasp"
        frames_by_split[split][frame_index] = relaxed_atoms
    outputs = []
    for split in split_names:
        write(f"{output_prefix}_{split}.traj", frames_by_split[split])
        outputs.append(f"{output_prefix}_{split}.traj")
        print(
            f"{name}: {split} {len(frames_by_split[split])} frames -> {output_prefix}_{split}.traj", flush=True)
    return outputs


def label_vasp(output_dir, inputs, *, name, **vasp_settings):
    """VASP single point of every frame, one SLURM job per frame; requires SCF convergence.

    Returns one <name>_<split>.xyz per input: extxyz with info["dft_energy"] (energy σ->0) and arrays["dft_forces"],
    without constraints. Also writes <name>_<split>_constraints.traj (same frames with constraints).
    """
    output_prefix = os.path.join(output_dir, name)
    split_names = unique_split_names(inputs)
    frames_to_label = [(split, frame_index, frame) for split_path, split in zip(inputs, split_names)
                       for frame_index, frame in enumerate(read(split_path, index=":"))]
    frame_dirs = vasp_jobs.run_vasp_frames({"name": name, **vasp_settings}, output_dir, frames_to_label,
                                           vasp_jobs.SCF_CONVERGED_MARKER, continue_from_contcar=False)
    labeled_frames_by_split = {split: [] for split in split_names}
    constrained_frames_by_split = {split: [] for split in split_names}
    for (split, _, _), frame_dir in zip(frames_to_label, frame_dirs):
        labeled_atoms = read(os.path.join(
            frame_dir, "OUTCAR"), format="vasp-out")
        # ASE's "energy" from an OUTCAR is the energy extrapolated to σ->0, not TOTEN
        labeled_atoms.info["dft_energy"] = labeled_atoms.get_potential_energy()
        # real forces on fixed atoms, not zeros
        labeled_atoms.arrays["dft_forces"] = labeled_atoms.get_forces(
            apply_constraint=False)
        # without a calculator, extxyz writes only the dft_* keys, not "energy" and "forces"
        labeled_atoms.calc = None
        constrained_frames_by_split[split].append(labeled_atoms.copy())
        # no constraints, so extxyz writes no move_mask column
        labeled_atoms.set_constraint()
        labeled_frames_by_split[split].append(labeled_atoms)
    outputs = []
    for split in split_names:
        with open(f"{output_prefix}_{split}.xyz", "w") as output_file:
            write(output_file, labeled_frames_by_split[split], format="extxyz")
            output_file.flush()
            os.fsync(output_file.fileno())
        outputs.append(f"{output_prefix}_{split}.xyz")
        write(f"{output_prefix}_{split}_constraints.traj",
              constrained_frames_by_split[split])
        print(
            f"{name}: {split} {len(labeled_frames_by_split[split])} labeled frames -> {output_prefix}_{split}.xyz", flush=True)
    return outputs


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


def opes_minimum(output_dir, inputs, *, name, temperature_K, colvar_stride, colvar_file="COLVAR", state_file="STATE",
                 fes_bins=None):
    """Frame of an OPES trajectory closest in CV space to the minimum of the FES reconstructed from STATE.

    STATE and COLVAR are read from the directory of the trajectory; colvar_stride = COLVAR rows per trajectory frame.
    Returns [<output_dir>/<name>.traj]; also writes <name>_fes.dat.
    """
    if len(inputs) != 1:
        raise ValueError(
            f"opes_minimum needs exactly one trajectory, got {inputs}")
    output_prefix = os.path.join(output_dir, name)
    opes_run_dir = os.path.dirname(inputs[0])

    fes_file = f"{output_prefix}_fes.dat"
    # FES_from_State.py takes kT in the energy unit of the STATE file, eV here
    fes_command = [sys.executable, FES_SCRIPT, "--state", os.path.join(opes_run_dir, state_file), "--outfile", fes_file,
                   "--kt", str(temperature_K * units.kB)]
    if fes_bins is not None:
        fes_command += ["--bin", str(fes_bins)]
    subprocess.run(fes_command, check=True)

    fes_column_names, fes_table = read_plumed_table(fes_file)
    # the last FES column is the free energy, the others are the CVs
    cv_names = fes_column_names[:-1]
    fes_minimum_cv_values = fes_table[np.argmin(fes_table[:, -1]), :-1]

    colvar_column_names, colvar_table = read_plumed_table(
        os.path.join(opes_run_dir, colvar_file))
    cvs_missing_in_colvar = set(cv_names) - set(colvar_column_names)
    if cvs_missing_in_colvar:
        raise ValueError(f"CVs {sorted(cvs_missing_in_colvar)} of the FES are not in {colvar_file} "
                         f"fields {colvar_column_names}")
    cv_values_per_frame = colvar_table[::colvar_stride, [
        colvar_column_names.index(cv_name) for cv_name in cv_names]]

    opes_trajectory = Trajectory(inputs[0])
    if len(cv_values_per_frame) != len(opes_trajectory):
        raise ValueError(f"{len(colvar_table)} COLVAR rows with colvar_stride {colvar_stride} give "
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
    return [f"{output_prefix}.traj"]


def analyze(output_dir, inputs, *, name, write_interval_fs, equilibration_ps, cache_dir, species, analysis_stride=1,
            breakpoints_ps=[], use_global_msd=False, rigid_body_check={"every_n_frames": 500, "rel_tolerance": 1e-6},
            stride=10, cutoff_factor=1.1):
    """Diffusion and coordination analysis of one trajectory; returns [<output_dir>/<name>_diffusion.csv, cached npz]."""
    if len(inputs) != 1:
        raise ValueError(f"analyze needs exactly one trajectory, got {inputs}")
    diffusion_csv = diffusion_analysis(
        inputs[0], os.path.join(output_dir, name), write_interval_fs=write_interval_fs, equilibration_ps=equilibration_ps,
        cache_dir=cache_dir, species=species, analysis_stride=analysis_stride, breakpoints_ps=breakpoints_ps,
        use_global_msd=use_global_msd, rigid_body_check=rigid_body_check)
    coordination_npz = coordination_analysis(
        inputs[0], write_interval_fs=write_interval_fs, equilibration_ps=equilibration_ps, cache_dir=cache_dir,
        stride=stride, cutoff_factor=cutoff_factor)
    return [diffusion_csv, coordination_npz]


def plot(output_dir, inputs, *, name, regimes={}, trend_regimes={}, highlights=[], coordination_files={}):
    """Diffusion and structure plots of all systems from <system>_diffusion.csv and coordination npz inputs.

    The system of an npz input is its stored system_name; coordination_files ({system: npz path}) adds or replaces entries.
    Returns the paths of all plots and of cn_correlations.csv.
    """
    unknown_inputs = [input_path for input_path in inputs if not input_path.endswith((".csv", ".npz"))]
    if unknown_inputs:
        raise ValueError(
            f"inputs must be .csv or .npz files, got {unknown_inputs}")
    npz_path_by_system = coordination_files_by_system(
        [input_path for input_path in inputs if input_path.endswith(".npz")], coordination_files)
    diffusion_results = load_diffusion_results(
        [input_path for input_path in inputs if input_path.endswith(".csv")])
    with plt.rc_context(PLOT_STYLE):
        plot_paths = plot_overviews(
            diffusion_results, regimes, trend_regimes, output_dir)
        plot_paths += plot_highlights(diffusion_results,
                                      regimes, highlights, output_dir)
        plot_paths += plot_diffusion_ratios(
            diffusion_results, regimes, trend_regimes, output_dir)
        if npz_path_by_system:
            coordination_means = load_coordination_means(npz_path_by_system)
            plot_paths += plot_coordination_trends(diffusion_results, coordination_means, regimes, trend_regimes,
                                                   output_dir)
            plot_paths.append(write_coordination_correlations(
                coordination_means, output_dir))
            plot_paths += plot_pair_distance_peaks(npz_path_by_system, output_dir)
            plot_paths += plot_metal_atoms_per_c(npz_path_by_system, output_dir)
            plot_paths += plot_com_distance_traces(npz_path_by_system, output_dir)
            plot_paths += plot_com_distance_densities(npz_path_by_system, output_dir)
    return plot_paths
