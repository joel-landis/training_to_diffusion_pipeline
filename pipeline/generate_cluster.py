import os
import sys

import numpy as np
from ase import Atoms
from ase.build import sort
from ase.data import atomic_numbers, covalent_radii
from ase.io import write
from mace.calculators import mace_mp

from pipeline.mlip_dynamics import relax_with_bfgs, run_bussi_md
from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES, mace_calculator_kwargs


def build_dispersed_cluster(symbols, radius, c_c_separation, cov_scale, max_attempts, random_number_generator):
    """Place atoms one by one at uniform random points in a sphere, rejecting points too close to placed atoms.

    Minimum distance: c_c_separation for C-C pairs, otherwise cov_scale * (sum of the two covalent radii); lengths in Å.
    Raises after max_attempts rejected points for one atom. Returns Atoms without cell, centred at the origin.
    """
    placed_positions = []
    for atom_index, symbol in enumerate(symbols):
        new_atom_covalent_radius = covalent_radii[atomic_numbers[symbol]]
        minimum_distances_to_placed_atoms = np.array([
            c_c_separation if symbol == "C" and placed_symbol == "C"
            else cov_scale * (new_atom_covalent_radius + covalent_radii[atomic_numbers[placed_symbol]])
            for placed_symbol in symbols[:atom_index]
        ])
        for _ in range(max_attempts):
            radial_uniform, polar_uniform, azimuthal_uniform = random_number_generator.random(
                3)
            # the cube root gives a uniform density in volume, arccos(2v - 1) a uniform density on the sphere
            radial_distance = radius * radial_uniform ** (1.0 / 3.0)
            polar_angle = np.arccos(2.0 * polar_uniform - 1.0)
            azimuthal_angle = 2.0 * np.pi * azimuthal_uniform
            candidate_position = radial_distance * np.array([np.sin(polar_angle) * np.cos(azimuthal_angle),
                                                             np.sin(
                                                                 polar_angle) * np.sin(azimuthal_angle),
                                                             np.cos(polar_angle)])
            if atom_index == 0 or np.all(np.linalg.norm(np.array(placed_positions) - candidate_position, axis=1)
                                         >= minimum_distances_to_placed_atoms):
                placed_positions.append(candidate_position)
                break
        else:
            raise RuntimeError(
                f"Failed to place {symbol} after {max_attempts} attempts; increase radius or decrease c_c_separation")
    return Atoms(symbols, placed_positions)


def run(settings, output_dir, inputs):
    """Build one cluster: random placement, BFGS relaxation, sorting by Z, Bussi equilibration.

    Returns [<output_dir>/<name>.traj], the equilibrated cluster centred with "vacuum" Å on each side.
    """
    merged_settings, output_prefix = write_merged_settings(
        STAGES["generate_cluster"], settings, output_dir, inputs)
    mace_calculator = mace_mp(**mace_calculator_kwargs(merged_settings))

    print(f"cluster: composition {merged_settings['composition']}, radius {merged_settings['radius']} A, "
          f"seed {merged_settings['seed']}", flush=True)
    print(
        f"cluster: calculator {mace_calculator_kwargs(merged_settings)}", flush=True)

    random_number_generator = np.random.default_rng(merged_settings["seed"])
    shuffled_symbols = [element for element,
                        count in merged_settings["composition"].items() for _ in range(count)]
    random_number_generator.shuffle(shuffled_symbols)

    cluster = build_dispersed_cluster(shuffled_symbols, merged_settings["radius"], merged_settings["c_c_separation"],
                                      merged_settings["cov_scale"], merged_settings["max_attempts"], random_number_generator)
    cluster.center(vacuum=merged_settings["vacuum"])
    write(f"{output_prefix}_pre_opt.xyz", cluster, format="xyz")
    bfgs_converged, bfgs_steps, final_max_force = relax_with_bfgs(
        cluster, mace_calculator, merged_settings["fmax"], merged_settings["opt_max_steps"], output_prefix, f"{merged_settings['name']} BFGS")
    print(f"BFGS converged to fmax={merged_settings['fmax']}: {bfgs_converged} ({bfgs_steps} steps, "
          f"final fmax {final_max_force:.3f} eV/A)", flush=True)

    sorted_cluster = sort(cluster, tags=cluster.numbers)
    # BFGS shrinks the cluster; re-centre to keep the vacuum
    sorted_cluster.center(vacuum=merged_settings["vacuum"])
    write(f"{output_prefix}_relaxed.traj", sorted_cluster)
    run_bussi_md(sorted_cluster, mace_calculator, temperature_K=merged_settings["temperature_K"], timestep_fs=merged_settings["timestep_fs"],
                 taut_fs=merged_settings["taut_fs"], nsteps=merged_settings["equilibration_nsteps"],
                 write_interval_fs=merged_settings["write_interval_fs"], log_interval_fs=merged_settings["log_interval_fs"],
                 remove_rigid_body_interval_fs=merged_settings["remove_rigid_body_interval_fs"],
                 random_number_generator=random_number_generator, output_prefix=f"{output_prefix}_equilibration",
                 progress_label=f"{merged_settings['name']} equilibration")
    sorted_cluster.center(vacuum=merged_settings["vacuum"])
    write(f"{output_prefix}.traj", sorted_cluster)
    print(f"cluster: wrote {output_prefix}_pre_opt.xyz, {output_prefix}_opt.traj, {output_prefix}_opt.log, {output_prefix}_relaxed.traj, "
          f"{output_prefix}_equilibration.traj, {output_prefix}_equilibration.log, {output_prefix}.traj", flush=True)
    outputs = [f"{output_prefix}.traj"]
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(
        sys.argv[1], "generate_cluster", sys.argv[2:])
    raise_on_problems("generate_cluster", settings, inputs)
    run(settings, ".", inputs)
