import math
import os
from itertools import groupby
from pathlib import Path

import numpy as np
from ase.build import sort
from ase.io import write

from pipeline import slurm_jobs
from pipeline.config import load_site_config
from pipeline.io_utils import write_csv, write_text

SCF_CONVERGED_MARKER = "aborting loop because EDIFF is reached"
RELAX_CONVERGED_MARKER = "reached required accuracy - stopping structural energy minimisation"
VASP_REQUIRED = ("incar", "potcars", "ntasks", "time")
VASP_DEFAULTS = {
    "kpoints": None,
    "sbatch_args": [],
    "mem_per_cpu": "2G",
    "sync_interval_s": 60,
    "poll_interval_s": 60,
}


def format_incar_value(value):
    """INCAR text for a TOML value: bool -> .TRUE./.FALSE., list -> space-joined, anything else -> str()."""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, list):
        return " ".join(str(list_element) for list_element in value)
    return str(value)


def ncore_for(ntasks):
    """NCORE for ntasks MPI ranks: ntasks/4 if divisible, else ntasks/5 if divisible, else ceil(ntasks/4)."""
    for divisor in (4, 5):
        if ntasks % divisor == 0:
            return ntasks // divisor
    return max(1, math.ceil(ntasks / 4))


def prepare_incar(vasp_settings):
    """INCAR tags of the stage: keys of incar uppercased, NCORE from ncore_for(ntasks) unless given.

    Raises if both an explicit kpoints mesh and KSPACING are set.
    """
    incar_tags = {tag.upper(): value for tag,
                  value in vasp_settings["incar"].items()}
    if "NCORE" not in incar_tags:
        incar_tags["NCORE"] = ncore_for(vasp_settings["ntasks"])
    if vasp_settings["ntasks"] % 4:
        print(
            f"{vasp_settings['name']}: ntasks {vasp_settings['ntasks']} is not divisible by 4; NCORE = {incar_tags['NCORE']}", flush=True)
    if vasp_settings["kpoints"] is not None and "KSPACING" in incar_tags:
        raise ValueError(
            "both 'kpoints' and INCAR KSPACING given; VASP ignores KSPACING when a KPOINTS file is present")
    return incar_tags


def kpoint_mesh(atoms, incar_tags, explicit_kpoints):
    """k-point mesh VASP will use for atoms, and whether it is Γ-centred.

    Without explicit_kpoints the mesh follows VASP's rule N_i = max(1, ceil(|b_i| 2π / KSPACING)), with KSPACING default 0.5
    and KGAMMA default true.
    Returns ([n1, n2, n3], gamma_centred).
    """
    if explicit_kpoints is not None:
        return list(explicit_kpoints["mesh"]), explicit_kpoints["gamma"]
    gamma_centred = incar_tags.get("KGAMMA", True)
    # a string such as "F" would be truthy and select the Γ-only executable
    if not isinstance(gamma_centred, bool):
        raise TypeError(
            f"KGAMMA must be a TOML boolean, got {gamma_centred!r}")
    kspacing = incar_tags.get("KSPACING", 0.5)
    # ASE's reciprocal cell has no 2π factor
    return [max(1, math.ceil(np.linalg.norm(reciprocal_vector) * 2 * math.pi / kspacing))
            for reciprocal_vector in atoms.cell.reciprocal()], gamma_centred


def vasp_job_script(vasp_settings, job_name, vasp_executable, files_copied_to_scratch, success_marker, continue_from_contcar):
    """job.sh text for one VASP frame.

    If success_marker is missing from OUTCAR, the job writes failure_<jobid>.txt and exits 1; with continue_from_contcar the
    CONTCAR becomes the POSCAR of the next submission.
    """
    job_script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={vasp_settings['ntasks']}",
        f"#SBATCH --time={vasp_settings['time']}",
        f"#SBATCH --mem-per-cpu={vasp_settings['mem_per_cpu']}",
        "#SBATCH --output=vasp_%j.out",
        "#SBATCH --error=vasp_%j.err",
    ]
    job_script_lines += [
        f"#SBATCH {extra_sbatch_line}" for extra_sbatch_line in vasp_settings["sbatch_args"]]
    job_script_lines.append("set -euo pipefail")
    job_script_lines += vasp_settings["job_setup"]
    # run in node-local scratch; rsync back periodically and on exit, also on failure
    job_script_lines += [
        "export OMP_NUM_THREADS=1",
        'SUBMIT_DIR="$SLURM_SUBMIT_DIR"',
        'SCRATCH_DIR="${TMPDIR:?TMPDIR not set}/$SLURM_JOB_ID"',
        'mkdir -p "$SCRATCH_DIR"',
        f'cp {" ".join(files_copied_to_scratch)} "$SCRATCH_DIR"/',
        'cd "$SCRATCH_DIR"',
        f'sync_back_loop() {{ while sleep {vasp_settings["sync_interval_s"]}; do rsync -a --update "$SCRATCH_DIR"/ "$SUBMIT_DIR"/; done; }}',
        "sync_back_loop &",
        "SYNC_PID=$!",
        'final_sync() { kill "$SYNC_PID"; rsync -a "$SCRATCH_DIR"/ "$SUBMIT_DIR"/; }',
        "trap final_sync EXIT",
        f"srun {vasp_executable} > vasp.out",
        # ASE reads CONTCAR with the OUTCAR and fails on the hashed POTCAR names VASP writes to line 6
        'sed -i "6s#.*#$(sed -n 6p POSCAR)#" CONTCAR',
        # remove the trap first so final_sync does not run a second time on exit
        "trap - EXIT",
        "final_sync",
        'cd "$SUBMIT_DIR"',
    ]
    job_script_lines.append(f'if ! grep -q "{success_marker}" OUTCAR; then')
    if continue_from_contcar:
        # start.poscar is written only on the first failure, so it always holds the original start structure
        job_script_lines += [
            '    [ -e start.poscar ] || mv POSCAR start.poscar',
            '    cp CONTCAR "run_${SLURM_JOB_ID}.contcar"',
            '    mv CONTCAR POSCAR',
            '    echo "ionic relaxation not converged within NSW; POSCAR now holds the CONTCAR of job ${SLURM_JOB_ID}, '
            'the original is start.poscar" > "failure_${SLURM_JOB_ID}.txt"',
        ]
    else:
        job_script_lines.append(
            '    echo "SCF not converged within NELM" > "failure_${SLURM_JOB_ID}.txt"')
    job_script_lines += [
        '    sync "failure_${SLURM_JOB_ID}.txt"',
        '    cat "failure_${SLURM_JOB_ID}.txt" >&2',
        '    exit 1',
        'fi',
    ]
    job_script_lines += [
        "echo '[\"OUTCAR\"]' > outputs.json",
        "sync outputs.json",
    ]
    return "\n".join(job_script_lines) + "\n"


def write_frame_inputs(vasp_settings, incar_tags, frame_dir, atoms, job_name, success_marker, continue_from_contcar):
    """Write POSCAR, INCAR, POTCAR, KPOINTS (only with explicit kpoints) and job.sh for one frame into frame_dir.

    Constraints are written as selective dynamics. The executable is gamma_executable for a Γ-centred 1x1x1 mesh, otherwise
    std_executable.
    """
    # POSCAR species blocks and POTCAR order both come from this Z-sorted order, so they always match
    atoms = sort(atoms, tags=atoms.numbers)
    species_in_poscar_order = [element for element,
                               _ in groupby(atoms.get_chemical_symbols())]
    species_without_potcar = set(
        species_in_poscar_order) - set(vasp_settings["potcars"])
    if species_without_potcar:
        raise KeyError(
            f"no POTCAR mapping for {sorted(species_without_potcar)} in 'potcars'")
    os.makedirs(frame_dir, exist_ok=True)
    write(os.path.join(frame_dir, "POSCAR"), atoms, format="vasp", direct=True)
    write_text(os.path.join(frame_dir, "INCAR"), "".join(
        f"{tag} = {format_incar_value(value)}\n" for tag, value in incar_tags.items()))
    potcar_text = ""
    for element in species_in_poscar_order:
        potcar_text += Path(vasp_settings["potcar_dir"], vasp_settings["potcars"]
                            [element], "POTCAR").read_text().strip() + "\n"
    write_text(os.path.join(frame_dir, "POTCAR"), potcar_text)
    kpoint_mesh_size, gamma_centred = kpoint_mesh(
        atoms, incar_tags, vasp_settings["kpoints"])
    files_copied_to_scratch = ["INCAR", "POSCAR", "POTCAR"]
    if vasp_settings["kpoints"] is not None:
        write_text(os.path.join(frame_dir, "KPOINTS"),
                   f"Automatic mesh\n0\n{'Gamma' if gamma_centred else 'Monkhorst-Pack'}\n{' '.join(map(str, kpoint_mesh_size))}\n0 0 0\n")
        files_copied_to_scratch.append("KPOINTS")
    vasp_executable = vasp_settings["gamma_executable"] if kpoint_mesh_size == [
        1, 1, 1] and gamma_centred else vasp_settings["std_executable"]
    # job.sh is written last, so its presence means the frame inputs are complete
    write_text(os.path.join(frame_dir, "job.sh"),
               vasp_job_script(vasp_settings, job_name, vasp_executable, files_copied_to_scratch, success_marker, continue_from_contcar))


def run_vasp_frames(vasp_settings, output_dir, frames_to_run, success_marker, continue_from_contcar):
    """One VASP job per frame: write the inputs of every frame without a job.sh, submit, wait, and write
    <output_dir>/<name>_status.csv. Raises if any frame failed.

    frames_to_run: list of (split, frame_index, atoms); frame i of split s runs in <output_dir>/<s>/<i:05d>.
    Returns the frame directories in the order of frames_to_run.
    """
    site_config = load_site_config()
    vasp_settings = {**VASP_DEFAULTS, **vasp_settings, **
                     site_config["vasp"], "job_setup": site_config["job_setup"]}
    incar_tags = prepare_incar(vasp_settings)
    frame_dirs = []
    for split, frame_index, atoms in frames_to_run:
        frame_dir = os.path.join(output_dir, split, f"{frame_index:05d}")
        if not os.path.exists(os.path.join(frame_dir, "job.sh")):
            write_frame_inputs(vasp_settings, incar_tags, frame_dir, atoms, f"{vasp_settings['name']}_{split}_{frame_index}",
                               success_marker, continue_from_contcar)
        frame_dirs.append(frame_dir)
    print(
        f"{vasp_settings['name']}: {len(frame_dirs)} frames prepared in {output_dir}", flush=True)
    job_id_by_dir, failed_state_by_dir = slurm_jobs.submit_and_wait_for_jobs(
        vasp_settings["name"], frame_dirs, vasp_settings["poll_interval_s"])

    status_rows = [[split, frame_index, job_id_by_dir[frame_dir], failed_state_by_dir.get(frame_dir, "COMPLETED"),
                    failure_reason(frame_dir, job_id_by_dir[frame_dir], failed_state_by_dir[frame_dir])
                    if frame_dir in failed_state_by_dir else ""]
                   for (split, frame_index, _), frame_dir in zip(frames_to_run, frame_dirs)]
    status_path = os.path.join(
        output_dir, f"{vasp_settings['name']}_status.csv")
    write_csv(status_path, ["split", "frame_index",
              "job_id", "state", "reason"], status_rows)
    if failed_state_by_dir:
        raise RuntimeError(f"{vasp_settings['name']}: {len(failed_state_by_dir)} of {len(status_rows)} frames failed, "
                           f"see the reason column of {status_path}; rerun the same config to resubmit the failed frames")
    return frame_dirs


def failure_reason(frame_dir, job_id, job_state):
    """Why a frame's job failed: the text of failure_<job_id>.txt written by the job script, or the SLURM state if
    VASP ended before the convergence check (crash, walltime, cancellation)."""
    reason_file = Path(frame_dir, f"failure_{job_id}.txt")
    if reason_file.exists():
        return reason_file.read_text().strip()
    return f"SLURM state {job_state}; VASP ended before the convergence check, see vasp_{job_id}.err"
