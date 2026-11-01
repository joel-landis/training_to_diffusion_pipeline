import contextlib
import os
import re
import sys

import numpy as np
from ase import units
from ase.calculators.calculator import all_changes
from ase.calculators.plumed import Plumed
from ase.io import read
from mace.calculators import mace_mp

from pipeline.mlip_dynamics import run_bussi_md
from pipeline.checks import raise_on_problems
from pipeline.config import standalone_settings, write_merged_settings
from pipeline.io_utils import write_json
from pipeline.stage_specs import STAGES, mace_calculator_kwargs

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


def run(settings, output_dir, inputs):
    """Bussi MD from one start structure, optionally biased with a PLUMED template.

    The start structure's velocities are not used. PLUMED output files (COLVAR, STATE, ...) are written to output_dir.
    Returns [<output_dir>/<name>.traj].
    """
    if len(inputs) != 1:
        raise ValueError(
            f"run_md needs exactly one start structure, got {inputs}")
    merged_settings, output_prefix = write_merged_settings(
        STAGES["run_md"], settings, output_dir, inputs)
    mace_calculator = mace_mp(**mace_calculator_kwargs(merged_settings))

    start_atoms = read(
        inputs[0], format=merged_settings["start_structure_format"])
    print(
        f"md: start {inputs[0]} ({len(start_atoms)} atoms, {start_atoms.get_chemical_formula()}), seed {merged_settings['seed']}", flush=True)
    print(
        f"md: calculator {mace_calculator_kwargs(merged_settings)}, plumed {merged_settings['plumed']}", flush=True)
    # output_prefix is made absolute because the PLUMED branch changes the working directory
    bussi_kwargs = dict(temperature_K=merged_settings["temperature_K"], timestep_fs=merged_settings["timestep_fs"],
                        taut_fs=merged_settings["taut_fs"], nsteps=merged_settings["nsteps"],
                        write_interval_fs=merged_settings[
                            "write_interval_fs"], log_interval_fs=merged_settings["log_interval_fs"],
                        remove_rigid_body_interval_fs=merged_settings["remove_rigid_body_interval_fs"],
                        random_number_generator=np.random.default_rng(
                            merged_settings["seed"]),
                        output_prefix=os.path.abspath(output_prefix), progress_label=merged_settings["name"])

    if merged_settings["plumed"] is None:
        run_bussi_md(start_atoms, mace_calculator, **bussi_kwargs)
    else:
        plumed_lines = plumed_input_lines(
            merged_settings["plumed"], start_atoms)
        plumed_timestep = merged_settings["timestep_fs"] * units.fs
        plumed_kt_ev = merged_settings["temperature_K"] * units.kB
        # PLUMED writes into the working directory; leaving the with block finalizes PLUMED
        with contextlib.chdir(output_dir), ScalarEnergyPlumed(calc=mace_calculator, input=plumed_lines, timestep=plumed_timestep,
                                                              atoms=start_atoms, kT=plumed_kt_ev) as plumed_biased_calculator:
            run_bussi_md(start_atoms, plumed_biased_calculator, **bussi_kwargs)
    print(
        f"md: wrote {output_prefix}.traj, {output_prefix}.log, {output_prefix}_settings.json", flush=True)
    outputs = [f"{output_prefix}.traj"]
    write_json(os.path.join(output_dir, "outputs.json"), outputs)
    return outputs


if __name__ == "__main__":
    settings, inputs = standalone_settings(sys.argv[1], "run_md", sys.argv[2:])
    raise_on_problems("run_md", settings, inputs)
    run(settings, ".", inputs)
