import contextlib
import os
import re

import ase
import numpy as np
from packaging.version import Version
from ase import Atoms, units
from ase.calculators.calculator import all_changes
from ase.calculators.plumed import Plumed
from ase.data import atomic_numbers, covalent_radii
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from tqdm import tqdm

ASE_MAX_VERSION_CHECKED = "3.30.0b1"
if Version(ase.__version__) > Version(ASE_MAX_VERSION_CHECKED):
    raise RuntimeError(
        f"ASE {ase.__version__} > {ASE_MAX_VERSION_CHECKED}; re-check the Bussi ndof patch")

ELEMENT_PLACEHOLDER = re.compile(r"@([A-Z][a-z]?)@")


class ScalarEnergyPlumed(Plumed):
    """ASE's Plumed calculator, returning the energy as a float.

    ASE adds PLUMED's bias, a shape-(1,) array, to the energy, so results["energy"] becomes an array; MDLogger and the
    trajectory then store an array instead of a number.
    """

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.results["energy"].item()


def plumed_input_lines(template_path, atoms):
    """PLUMED input lines from a template, with @El@ replaced by the comma-joined 1-based indices of element El in atoms.

    Blank lines and lines starting with # are dropped. Raises if a placeholder names an element that is not in atoms.
    """
    symbols = atoms.get_chemical_symbols()

    def indices_of_element(placeholder_match):
        # PLUMED numbers atoms from 1, ASE from 0
        one_based_indices = [str(atom_index + 1) for atom_index, symbol in enumerate(
            symbols) if symbol == placeholder_match.group(1)]
        if not one_based_indices:
            raise ValueError(f"{template_path}: placeholder {placeholder_match.group(0)} but no {placeholder_match.group(1)} "
                             f"in {atoms.get_chemical_formula()}")
        return ",".join(one_based_indices)

    with open(template_path) as template_file:
        return [ELEMENT_PLACEHOLDER.sub(indices_of_element, line.strip()) for line in template_file
                if line.strip() and not line.lstrip().startswith("#")]


@contextlib.contextmanager
def md_calculator(mace_calculator, plumed_template, atoms, timestep_fs, temperature_K, output_dir):
    """The MACE calculator, or with a PLUMED template the MACE calculator biased by PLUMED."""
    if plumed_template is None:
        yield mace_calculator
        return
    # read before the chdir below, so a relative template path refers to the working directory of the run
    plumed_lines = plumed_input_lines(plumed_template, atoms)
    # PLUMED writes into the working directory; leaving the with block finalizes PLUMED
    with contextlib.chdir(output_dir), ScalarEnergyPlumed(
            calc=mace_calculator, input=plumed_lines,
            timestep=timestep_fs * units.fs, atoms=atoms, kT=temperature_K * units.kB) as plumed_biased_calculator:
        yield plumed_biased_calculator


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



def relax_with_bfgs(atoms, mace_calculator, fmax, max_steps, output_prefix, progress_label):
    """Relax atoms in place with BFGS; stopping unconverged at max_steps is not an error.

    Writes <output_prefix>_opt.traj and _opt.log.
    Returns (converged, number of steps, final max force in eV/Å).
    """
    atoms.calc = mace_calculator
    bfgs_optimizer = BFGS(
        atoms, trajectory=f"{output_prefix}_opt.traj", logfile=f"{output_prefix}_opt.log")
    progress_bar = tqdm(total=max_steps, unit="step", desc=progress_label)
    bfgs_optimizer.attach(lambda: progress_bar.update(
        bfgs_optimizer.nsteps - progress_bar.n))
    converged = bfgs_optimizer.run(fmax=fmax, steps=max_steps)
    progress_bar.close()
    final_max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
    return converged, bfgs_optimizer.nsteps, final_max_force


def run_bussi_md(atoms, mace_calculator, temperature_K, timestep_fs, taut_fs, nsteps, write_interval_fs, log_interval_fs,
                 remove_rigid_body_interval_fs, random_number_generator, output_prefix, progress_label, plumed_template=None):
    """Bussi NVT MD of an isolated system, in place, with fresh Maxwell-Boltzmann velocities at temperature_K.

    Centre-of-mass translation and rotation are removed every remove_rigid_body_interval_fs. random_number_generator draws the
    velocities and the thermostat noise. Writes <output_prefix>.traj and <output_prefix>.log. With plumed_template the MD is
    biased by PLUMED, whose output files (COLVAR, STATE, ...) go next to <output_prefix>.
    """
    # absolute, because the PLUMED case changes the working directory
    output_prefix = os.path.abspath(output_prefix)
    with md_calculator(mace_calculator, plumed_template, atoms, timestep_fs, temperature_K,
                       os.path.dirname(output_prefix)) as calculator:
        atoms.calc = calculator

        def remove_com_translation_and_rotation():
            Stationary(atoms)
            ZeroRotation(atoms)

        MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature_K, rng=random_number_generator)
        remove_com_translation_and_rotation()

        bussi_dynamics = Bussi(atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K, taut=taut_fs * units.fs,
                               rng=random_number_generator)

        # Bussi counts 3N dof, but the 6 rigid-body modes are removed; constraints are already counted by ASE
        ase_thermostat_ndof = bussi_dynamics.ndof
        bussi_dynamics.ndof = atoms.get_number_of_degrees_of_freedom() - 6
        bussi_dynamics.target_kinetic_energy = 0.5 * \
            bussi_dynamics.temp * bussi_dynamics.ndof
        print(f"{progress_label}: T {temperature_K} K, dt {timestep_fs} fs, taut {taut_fs} fs, nsteps {nsteps} "
              f"({nsteps * timestep_fs / 1000:g} ps), thermostat ndof ASE {ase_thermostat_ndof} -> patched {bussi_dynamics.ndof}", flush=True)

        progress_bar = tqdm(total=nsteps, unit="step", desc=progress_label)

        def show_internal_temperature():
            internal_temperature_K = 2 * atoms.get_kinetic_energy() / \
                (bussi_dynamics.ndof * units.kB)
            progress_bar.set_postfix(
                T_int=f"{internal_temperature_K:.0f} K", refresh=False)

        trajectory_writer = Trajectory(f"{output_prefix}.traj", "w", atoms)
        log_interval_steps = round(log_interval_fs / timestep_fs)
        bussi_dynamics.attach(MDLogger(
            bussi_dynamics, atoms, f"{output_prefix}.log", mode="w"), interval=log_interval_steps)
        bussi_dynamics.attach(trajectory_writer, interval=round(
            write_interval_fs / timestep_fs))
        bussi_dynamics.attach(remove_com_translation_and_rotation, interval=round(
            remove_rigid_body_interval_fs / timestep_fs))
        bussi_dynamics.attach(lambda: progress_bar.update(
            bussi_dynamics.nsteps - progress_bar.n))
        bussi_dynamics.attach(show_internal_temperature,
                              interval=log_interval_steps)
        bussi_dynamics.run(nsteps)
        progress_bar.close()
        trajectory_writer.close()
