import os

import matplotlib.pyplot as plt
import numpy as np
import tidynamics
from ase.io.trajectory import Trajectory
from scipy.signal import correlate
from scipy.spatial.transform import Rotation
from scipy.stats import linregress
from tqdm import tqdm

from pipeline.analysis_utils import cache_key, frames_from_ps, load_or_compute_cached, trajectory_cache_fields
from pipeline.io_utils import write_csv
from pipeline.plot_utils import savefig_fsync


def inertia_tensor(masses, positions_relative_to_center):
    """Inertia tensor sum_a m_a (|r_a|^2 I - r_a r_a^T), a (3, 3) array; positions relative to the centre of mass."""
    return np.einsum("a,aij->ij", masses,
                     np.eye(3) * np.einsum("ai,ai->a", positions_relative_to_center,
                                           positions_relative_to_center)[:, None, None]
                     - np.einsum("ai,aj->aij", positions_relative_to_center, positions_relative_to_center))


def angular_momentum(masses, positions_relative_to_center, velocities):
    """Total angular momentum sum_a m_a r_a x v_a, a (3,) array; positions and velocities are (n_atoms, 3)."""
    return np.sum(masses[:, None] * np.cross(positions_relative_to_center, velocities), axis=0)


def rigid_body_residuals(masses, positions_relative_to_center, velocities):
    """Remaining linear and angular momentum after rigid-body removal, each relative to its scale sum_a m_a |v_a| and
    sum_a m_a |r_a| |v_a|; both should be ~0."""
    momentum_scale = np.sum(masses[:, None] * np.abs(velocities))
    angular_momentum_scale = np.sum(masses * np.linalg.norm(
        positions_relative_to_center, axis=1) * np.linalg.norm(velocities, axis=1))
    relative_linear_momentum = np.linalg.norm(np.sum(
        masses[:, None] * velocities, axis=0)) / momentum_scale if momentum_scale > 0 else 0.0
    relative_angular_momentum = (np.linalg.norm(angular_momentum(masses, positions_relative_to_center, velocities)) / angular_momentum_scale
                                 if angular_momentum_scale > 0 else 0.0)
    return relative_linear_momentum, relative_angular_momentum


def remove_rigid_body_motion_and_align(trajectory_path, first_frame_index, stride, check_every_n_frames, check_rel_tolerance):
    """Positions and velocities of the cluster's internal motion, from first_frame_index on.

    Per frame: centre of mass moved to the box centre, centre-of-mass velocity and angular velocity removed from the
    velocities, then positions and velocities rotated onto the previous frame (mass-weighted best fit), so the cluster
    neither translates nor rotates over the trajectory. Every stride-th frame is kept; the alignment uses every frame.
    Every check_every_n_frames-th frame, raises if rigid_body_residuals exceed check_rel_tolerance.
    Returns {"positions": (n_frames, n_atoms, 3) Å, "velocities": (n_frames, n_atoms, 3), "energies": (n_frames,) eV,
    "symbols": (n_atoms,)}.
    """
    trajectory = Trajectory(trajectory_path)
    if first_frame_index >= len(trajectory):
        raise ValueError(
            f"equilibration cut at frame {first_frame_index} but {trajectory_path} has {len(trajectory)} frames")
    kept_frame_indices = range(first_frame_index, len(trajectory), stride)
    first_frame = trajectory[first_frame_index]
    masses = first_frame.get_masses()
    box_center = first_frame.get_cell().lengths() / 2.0
    aligned_positions = np.empty(
        (len(kept_frame_indices), len(first_frame), 3))
    internal_velocities = np.empty_like(aligned_positions)
    potential_energies = np.empty(len(kept_frame_indices))

    previous_aligned_positions = first_frame.get_positions(
    ) + box_center - masses @ first_frame.get_positions() / masses.sum()
    previous_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    worst_relative_momentum = worst_relative_angular_momentum = 0.0
    next_output_row = 0
    for frame_index in tqdm(range(first_frame_index, len(trajectory)), desc="preprocessing"):
        frame = trajectory[frame_index]
        potential_energy = frame.get_potential_energy()
        centred_positions = frame.get_positions() + box_center - \
            masses @ frame.get_positions() / masses.sum()
        velocities = frame.get_velocities()
        velocities = velocities - masses @ velocities / masses.sum()
        positions_relative_to_center = centred_positions - box_center
        # angular velocity ω = I^-1 L; subtracting ω x r removes the rotation from the velocities
        angular_velocity = np.linalg.solve(inertia_tensor(masses, positions_relative_to_center),
                                           angular_momentum(masses, positions_relative_to_center, velocities))
        velocities = velocities - \
            np.cross(angular_velocity, positions_relative_to_center)

        alignment_rotation, _ = Rotation.align_vectors(
            previous_aligned_positions - box_center, positions_relative_to_center, weights=masses)
        alignment_quaternion = alignment_rotation.as_quat()
        # q and -q are the same rotation; keep the sign continuous with the previous frame
        if np.dot(previous_quaternion, alignment_quaternion) < 0.0:
            alignment_quaternion = -alignment_quaternion
            alignment_rotation = Rotation.from_quat(alignment_quaternion)
        centred_positions = alignment_rotation.apply(
            positions_relative_to_center) + box_center
        velocities = alignment_rotation.apply(velocities)

        if (frame_index - first_frame_index) % check_every_n_frames == 0:
            relative_linear_momentum, relative_angular_momentum = rigid_body_residuals(
                masses, centred_positions - box_center, velocities)
            worst_relative_momentum = max(
                worst_relative_momentum, relative_linear_momentum)
            worst_relative_angular_momentum = max(
                worst_relative_angular_momentum, relative_angular_momentum)
            if relative_linear_momentum > check_rel_tolerance or relative_angular_momentum > check_rel_tolerance:
                raise RuntimeError(f"rigid-body removal failed at frame {frame_index}: relative |P| {relative_linear_momentum:.3e}, "
                                   f"|L| {relative_angular_momentum:.3e}, tolerance {check_rel_tolerance:.3e}")
        if (frame_index - first_frame_index) % stride == 0:
            aligned_positions[next_output_row] = centred_positions
            internal_velocities[next_output_row] = velocities
            potential_energies[next_output_row] = potential_energy
            next_output_row += 1
        previous_aligned_positions, previous_quaternion = centred_positions, alignment_quaternion
    symbols = np.array(first_frame.get_chemical_symbols())
    trajectory.close()
    print(
        f"rigid-body check passed: worst relative |P| {worst_relative_momentum:.3e}, |L| {worst_relative_angular_momentum:.3e}", flush=True)
    return {"positions": aligned_positions, "velocities": internal_velocities, "energies": potential_energies, "symbols": symbols}


def velocity_autocorrelation(velocities, cutoff_frames):
    """Velocity autocorrelation summed over all atoms and components, averaged over time origins, for lags 0 to cutoff_frames - 1.

    velocities: (n_frames, n_atoms, 3) array.
    """
    velocity_components = velocities.reshape(velocities.shape[0], -1)
    n_frames = velocity_components.shape[0]
    summed_autocorrelation = np.zeros(n_frames)
    for component in range(velocity_components.shape[1]):
        # [n_frames - 1:] keeps the non-negative lags
        summed_autocorrelation += correlate(velocity_components[:, component],
                                            velocity_components[:, component], mode="full", method="fft")[n_frames - 1:]
    # lag k has n_frames - k time origins
    return (summed_autocorrelation / (n_frames - np.arange(n_frames)))[:min(cutoff_frames, n_frames)]


def correlation_time(normalised_vacf, epsilon, species):
    """Correlation time in frames: first lag after which |VACF| stays within epsilon up to the cutoff.

    normalised_vacf: VACF divided by its value at lag 0.
    Raises if the VACF is still above epsilon at the cutoff.
    """
    lags_above_epsilon = np.flatnonzero(np.abs(normalised_vacf) > epsilon)
    correlation_time_frames = int(
        lags_above_epsilon[-1]) + 1 if lags_above_epsilon.size else 0
    if correlation_time_frames >= len(normalised_vacf):
        raise RuntimeError(
            f"{species}: VACF stays above {epsilon} up to the cutoff of {len(normalised_vacf)} frames")
    return correlation_time_frames


def mean_squared_displacement(positions):
    """MSD for lags 0 to n_frames - 1, averaged over time origins and atoms; positions is (n_frames, n_atoms, 3) in Å."""
    return sum(tidynamics.msd(positions[:, atom_index, :]) for atom_index in range(positions.shape[1])) / positions.shape[1]


def blocked_mean_squared_displacement(positions, block_length_frames):
    """MSD of consecutive, non-overlapping blocks, averaged over atoms; trailing frames that do not fill a block are dropped.

    positions: (n_frames, n_atoms, 3) aligned positions in Å.
    Returns {"msd_blocks": (n_blocks, block_length_frames - 1) MSD for lags 1 to block_length_frames - 1,
             "block_starts": first frame of each block}.
    """
    if block_length_frames >= len(positions):
        raise ValueError(
            f"block length {block_length_frames} frames >= span length {len(positions)} frames")
    block_start_frames = np.arange(
        0, len(positions) - block_length_frames + 1, block_length_frames)
    return {"msd_blocks": np.array([mean_squared_displacement(positions[block_start:block_start + block_length_frames])[1:]
                                    for block_start in tqdm(block_start_frames, desc="blocked MSD")]),
            "block_starts": block_start_frames}


def fit_diffusion_coefficient(lag_frames, msd_values, fit_start_frame, fit_end_frame, dt_fs):
    """Linear fit of MSD(t) between fit_start_frame and fit_end_frame; D = slope / 6 (Einstein relation in 3D).

    msd_values in Å^2, dt_fs = time between frames.
    Returns (D in Å^2/ps, R^2, fit times in ps, fitted MSD at those times).
    """
    in_fit_window = (lag_frames >= fit_start_frame) & (
        lag_frames <= fit_end_frame)
    fit_times_ps = lag_frames[in_fit_window] * dt_fs * 1e-3
    linear_fit = linregress(fit_times_ps, msd_values[in_fit_window])
    return linear_fit.slope / 6.0, linear_fit.rvalue**2, fit_times_ps, linear_fit.slope * fit_times_ps + linear_fit.intercept


def plot_potential(frame_times_ps, potential_energies, breakpoints_ps, path):
    """E_pot over time with the breakpoints as dashed red lines, saved to path."""
    figure, axes = plt.subplots(figsize=(12, 6))
    axes.plot(frame_times_ps, potential_energies, color="tab:blue")
    for breakpoint_ps in breakpoints_ps:
        axes.axvline(breakpoint_ps, color="red", linestyle="--")
    axes.set_xlabel("Time (ps)")
    axes.set_ylabel(r"$E_{pot}$ (eV)")
    savefig_fsync(figure, path)


def plot_vacf(normalised_vacf, correlation_time_frames, dt_fs, output_prefix):
    """Three VACF plots with the correlation time marked: zoom on ±0.05, log-log of |VACF|, log-x.

    Saved as <output_prefix>_vacf_plateau.png, _vacf_full_loglog.png and _vacf_full_logx.png.
    """
    lag_times_ps = np.arange(len(normalised_vacf)) * dt_fs * 1e-3
    correlation_time_ps = correlation_time_frames * dt_fs * 1e-3
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(lag_times_ps, normalised_vacf, color="tab:red")
    axes.axhline(0.0, color="black", linestyle="--")
    axes.axvline(correlation_time_ps, color="tab:green", linestyle=":")
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel("Velocity correlation")
    axes.set_xlim(0.0, 2.0)
    axes.set_ylim(-0.05, 0.05)
    savefig_fsync(figure, f"{output_prefix}_vacf_plateau.png")

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(lag_times_ps[1:], np.abs(
        normalised_vacf[1:]), color="tab:red", linewidth=0.8)
    axes.axvline(correlation_time_ps, color="tab:green", linestyle=":")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel("|Velocity correlation|")
    savefig_fsync(figure, f"{output_prefix}_vacf_full_loglog.png")

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(lag_times_ps[1:], normalised_vacf[1:],
              color="tab:red", linewidth=0.8)
    axes.axhline(0.0, color="black", linestyle="--")
    axes.axvline(correlation_time_ps, color="tab:green", linestyle=":")
    axes.set_xscale("log")
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel("Velocity correlation")
    savefig_fsync(figure, f"{output_prefix}_vacf_full_logx.png")


def plot_blocks(msd_blocks, block_fit_curves, block_diffusion_coefficients, global_msd, global_fit, dt_fs, output_prefix):
    """Four plots of one species and window: block MSDs with mean and fit (and global MSD), histogram of the block D,
    β(t) = d ln MSD / d ln t per block, and all block MSDs.

    block_fit_curves: (fit times in ps, list of fitted MSD per block).
    global_msd, global_fit: MSD over the whole window and its fit_diffusion_coefficient result, or None.
    """
    lag_times_ps = np.arange(1, msd_blocks.shape[1] + 1) * dt_fs * 1e-3

    figure, axes = plt.subplots(figsize=(12, 6))
    if global_msd is not None:
        axes.plot(np.arange(1, len(global_msd) + 1) * dt_fs * 1e-3,
                  global_msd, color="black", linewidth=2, label="Global MSD")
        axes.plot(global_fit[2], global_fit[3],
                  color="tab:red", linestyle="--", label="Global fit")
    for block_msd_curve in msd_blocks:
        axes.plot(lag_times_ps, block_msd_curve, color="tab:blue", alpha=0.10)
    axes.plot(lag_times_ps, msd_blocks.mean(axis=0),
              color="tab:blue", linewidth=2, label="Mean blocked MSD")
    axes.plot(block_fit_curves[0], np.mean(block_fit_curves[1], axis=0),
              color="tab:orange", linestyle="--", label="Mean blocked fit")
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel(r"MSD ($\AA^2$)")
    axes.legend()
    savefig_fsync(figure, f"{output_prefix}_global_vs_blocked_msd.png")

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.hist(block_diffusion_coefficients,
              bins=20, color="tab:blue", alpha=0.8)
    axes.axvline(block_diffusion_coefficients.mean(),
                 color="black", linestyle="--", label="Blocked mean")
    if global_fit is not None:
        axes.axvline(global_fit[0], color="tab:red",
                     linestyle=":", label="Global")
    axes.set_xlabel(r"$D$ ($\AA^2$/ps)")
    axes.set_ylabel("Count")
    axes.legend()
    savefig_fsync(figure, f"{output_prefix}_blocked_d_histogram.png")

    # central differences; β = 1 is diffusive
    beta_curves = np.full_like(msd_blocks, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_curves[:, 1:-1] = np.log(msd_blocks[:, 2:] / msd_blocks[:, :-2]) / \
            np.log(lag_times_ps[2:] / lag_times_ps[:-2])
    figure, axes = plt.subplots(figsize=(12, 6))
    for block_beta_curve in beta_curves:
        axes.plot(lag_times_ps, block_beta_curve,
                  color="darkorange", alpha=0.08)
    axes.axhline(1.0, color="black", linestyle="--")
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel(r"$\beta(t)$")
    axes.set_ylim(-5, 5)
    savefig_fsync(figure, f"{output_prefix}_beta_curves_all_blocks.png")

    figure, axes = plt.subplots(figsize=(12, 6))
    for block_msd_curve in msd_blocks:
        axes.plot(lag_times_ps, block_msd_curve, color="tab:blue", alpha=0.12)
    axes.set_xlabel("Lag time (ps)")
    axes.set_ylabel(r"MSD ($\AA^2$)")
    savefig_fsync(figure, f"{output_prefix}_msd_curves_all_blocks.png")


def diffusion_analysis(trajectory_path, output_prefix, *, write_interval_fs, equilibration_ps, cache_dir, species,
                       analysis_stride, breakpoints_ps, use_global_msd, rigid_body_check):
    """Diffusion coefficient of each species from blocked MSD fits, for the whole trajectory and each piece between breakpoints.

    Intermediate arrays are cached in cache_dir, keyed by the parameters they depend on.
    Returns <output_prefix>_diffusion.csv; also writes the plots.
    """
    trajectory_path = os.path.abspath(trajectory_path)
    label = os.path.basename(output_prefix)
    os.makedirs(cache_dir, exist_ok=True)

    analysis_timestep_fs = write_interval_fs * \
        analysis_stride
    equilibration_cut_frames = frames_from_ps(
        equilibration_ps, write_interval_fs)
    # positions and velocities without rigid-body motion from the equilibration cut on, cached per trajectory and settings
    preprocessing_cache_fields = dict(**trajectory_cache_fields(trajectory_path), start_idx=equilibration_cut_frames, stride=analysis_stride,
                                      write_interval_fs=write_interval_fs, rigid_body_check=rigid_body_check)
    preprocessing_cache_key = cache_key(**preprocessing_cache_fields)
    preprocessed_cache_path = os.path.join(
        cache_dir, f"preprocessed_{preprocessing_cache_key}.npz")
    preprocessed = load_or_compute_cached(
        preprocessed_cache_path,
        lambda: remove_rigid_body_motion_and_align(trajectory_path, equilibration_cut_frames, analysis_stride,
                                                   rigid_body_check["every_n_frames"], rigid_body_check["rel_tolerance"]))
    n_analysed_frames = len(preprocessed["positions"])
    frame_times_ps = (equilibration_cut_frames + np.arange(n_analysed_frames) *
                      analysis_stride) * write_interval_fs * 1e-3
    print(f"{label}: {n_analysed_frames} frames from {frame_times_ps[0]:g} to {frame_times_ps[-1]:g} ps, "
          f"analysis dt {analysis_timestep_fs:g} fs", flush=True)

    # analysis windows: the whole trajectory ("all") and, with breakpoints, every piece between them
    breakpoint_frames = [int(np.searchsorted(frame_times_ps, breakpoint_ps))
                         for breakpoint_ps in breakpoints_ps]
    if any(not 0 < breakpoint_frame < n_analysed_frames for breakpoint_frame in breakpoint_frames) or breakpoint_frames != sorted(set(breakpoint_frames)):
        raise ValueError(f"breakpoints_ps {breakpoints_ps} must be increasing and inside "
                         f"({frame_times_ps[0]:g}, {frame_times_ps[-1]:g}) ps")
    piece_boundaries = [0] + breakpoint_frames + [n_analysed_frames]
    pieces = [(f"piece_{piece_number + 1}", piece_start, piece_end)
              for piece_number, (piece_start, piece_end) in enumerate(zip(piece_boundaries[:-1], piece_boundaries[1:]))]
    analysis_windows = [("all", 0, n_analysed_frames)] + \
        (pieces if breakpoint_frames else [])
    plot_potential(frame_times_ps, preprocessed["energies"],
                   breakpoints_ps, f"{output_prefix}_potential_energy.png")

    diffusion_rows = []
    for species, species_settings in species.items():
        species_atom_indices = np.flatnonzero(
            preprocessed["symbols"] == species)
        if species_atom_indices.size == 0:
            raise ValueError(f"no {species} atoms in {trajectory_path}")
        vacf_cache_key = cache_key(
            prep=preprocessing_cache_key, species=species, cutoff_ps=species_settings["vacf_cutoff_ps"])
        vacf_cache_path = os.path.join(
            cache_dir, f"vacf_{vacf_cache_key}.npz")
        species_velocities = preprocessed["velocities"][:,
                                                        species_atom_indices]
        vacf_cutoff_frames = frames_from_ps(
            species_settings["vacf_cutoff_ps"], analysis_timestep_fs)
        unnormalised_vacf = load_or_compute_cached(
            vacf_cache_path, lambda: {"vacf": velocity_autocorrelation(species_velocities, vacf_cutoff_frames)})["vacf"]
        normalised_vacf = unnormalised_vacf / unnormalised_vacf[0]
        # the correlation time tau from this species' VACF sets the block length
        correlation_time_frames = correlation_time(
            normalised_vacf, species_settings["vacf_epsilon"], species)
        plot_vacf(normalised_vacf, correlation_time_frames,
                  analysis_timestep_fs, f"{output_prefix}_{species}")
        # blocks many correlation times long, so the block D values are close to independent samples
        block_length_frames = int(
            species_settings["n_samples_per_block"] * correlation_time_frames)
        fit_start_frame = frames_from_ps(
            species_settings["fit_start_ps"], analysis_timestep_fs)
        fit_end_frame = fit_start_frame + \
            frames_from_ps(
                species_settings["fit_width_ps"], analysis_timestep_fs)
        if fit_end_frame >= block_length_frames:
            raise ValueError(f"{species}: fit window ends at {fit_end_frame} frames but blocks are {block_length_frames} frames "
                             f"({species_settings['n_samples_per_block']} x tau {correlation_time_frames})")

        for window_name, window_start, window_end in analysis_windows:
            species_positions = preprocessed["positions"][window_start:window_end,
                                                          species_atom_indices]
            block_msd_cache_key = cache_key(prep=preprocessing_cache_key, species=species, span=[window_start, window_end],
                                            block_len=block_length_frames)
            block_msd_cache_path = os.path.join(
                cache_dir, f"blocks_{block_msd_cache_key}.npz")
            block_msds = load_or_compute_cached(block_msd_cache_path, lambda: blocked_mean_squared_displacement(
                species_positions, block_length_frames))
            # D of every block from the fit window of its MSD; the spread over the blocks gives the SEM
            block_lag_frames = np.arange(1, block_length_frames)
            block_fits = [fit_diffusion_coefficient(block_lag_frames, block_msd_curve, fit_start_frame, fit_end_frame, analysis_timestep_fs)
                          for block_msd_curve in block_msds["msd_blocks"]]
            block_diffusion_coefficients = np.array(
                [block_fit[0] for block_fit in block_fits])
            block_r_squared = np.array([block_fit[1]
                                       for block_fit in block_fits])
            diffusion_coefficient_sem = (block_diffusion_coefficients.std(ddof=1) / np.sqrt(len(block_diffusion_coefficients))
                                         if len(block_diffusion_coefficients) > 1 else np.nan)
            global_msd = global_fit = None
            if use_global_msd:
                global_msd_cache_key = cache_key(
                    prep=preprocessing_cache_key, species=species, span=[window_start, window_end])
                global_msd_cache_path = os.path.join(
                    cache_dir, f"msd_{global_msd_cache_key}.npz")
                global_msd = load_or_compute_cached(global_msd_cache_path,
                                                    lambda: {"msd": mean_squared_displacement(species_positions)[1:]})["msd"]
                global_fit = fit_diffusion_coefficient(np.arange(1, len(global_msd) + 1), global_msd, fit_start_frame, fit_end_frame,
                                                       analysis_timestep_fs)
            plot_blocks(block_msds["msd_blocks"], (block_fits[0][2], [block_fit[3] for block_fit in block_fits]), block_diffusion_coefficients,
                        global_msd, global_fit, analysis_timestep_fs, f"{output_prefix}_{species}_{window_name}")
            # one row per window and species, in the column order of the header written below
            diffusion_rows.append([window_name, species, frame_times_ps[window_start], frame_times_ps[window_end - 1],
                                   block_diffusion_coefficients.mean(), diffusion_coefficient_sem, len(
                                       block_diffusion_coefficients),
                                   correlation_time_frames * analysis_timestep_fs *
                                   1e-3, block_length_frames * analysis_timestep_fs * 1e-3,
                                   fit_start_frame * analysis_timestep_fs *
                                   1e-3, fit_end_frame * analysis_timestep_fs * 1e-3,
                                   block_r_squared.mean(), block_r_squared.min(), block_r_squared.max(),
                                   "" if global_fit is None else global_fit[0]])
            print(f"{label} [{window_name} | {species}] D = {block_diffusion_coefficients.mean():.6g} "
                  f"+- {diffusion_coefficient_sem:.2g} A^2/ps from {len(block_diffusion_coefficients)} blocks, "
                  f"tau {correlation_time_frames * analysis_timestep_fs * 1e-3:.3f} ps, R^2 {block_r_squared.min():.3f}-{block_r_squared.max():.3f}"
                  + ("" if global_fit is None else f", global D {global_fit[0]:.6g}"), flush=True)

    write_csv(f"{output_prefix}_diffusion.csv", ["span", "species", "start_ps", "end_ps", "D_mean_A2_per_ps", "D_sem_A2_per_ps", "n_blocks",
                                                 "tau_ps", "block_len_ps", "fit_start_ps", "fit_end_ps", "r2_mean", "r2_min", "r2_max",
                                                 "D_global_A2_per_ps"], diffusion_rows)
    return f"{output_prefix}_diffusion.csv"
