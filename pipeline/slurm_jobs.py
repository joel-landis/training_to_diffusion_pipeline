import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline.config import load_site_config
from pipeline.io_utils import write_text

ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
DEFAULT_TIME = "24:00:00"
SACCT_ATTEMPTS = 10
SACCT_RETRY_INTERVAL_S = 60


def slurm_job_script(settings, job_dir, job_name, command):
    """job.sh text that runs command as a SLURM job.

    job_dir receives the SLURM logs; the job_setup lines of site_config.toml run before the command.
    """
    slurm_settings = settings["slurm"]
    job_dir_absolute = os.path.abspath(job_dir)
    job_script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={slurm_settings.get('cpus_per_task', 1)}",
        f"#SBATCH --mem-per-cpu={slurm_settings.get('mem_per_cpu', '6G')}",
        f"#SBATCH --time={slurm_settings.get('time', DEFAULT_TIME)}",
        # relative paths refer to the orchestrator's working directory, not job_dir
        f"#SBATCH --chdir={os.getcwd()}",
        f"#SBATCH --output={job_dir_absolute}/slurm_%j.out",
        f"#SBATCH --error={job_dir_absolute}/slurm_%j.err",
    ]
    if "gpus" in slurm_settings:
        job_script_lines.append(f"#SBATCH --gpus={slurm_settings['gpus']}")
    job_script_lines += [
        f"#SBATCH {extra_sbatch_line}" for extra_sbatch_line in slurm_settings.get("sbatch_args", [])]
    job_script_lines.append("set -euo pipefail")
    job_script_lines += load_site_config()["job_setup"]
    job_script_lines.append(command)
    return "\n".join(job_script_lines) + "\n"


def submit_job(job_dir):
    """Submit job_dir/job.sh with sbatch from inside job_dir and write the job ID to job_dir/job_id."""
    sbatch_result = subprocess.run(
        ["sbatch", "--parsable", "job.sh"], cwd=job_dir, capture_output=True, text=True)
    if sbatch_result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed in {job_dir}: {sbatch_result.stderr.strip()}")
    # --parsable prints "<id>" or "<id>;<cluster>" on multi-cluster setups
    write_text(os.path.join(job_dir, "job_id"),
               sbatch_result.stdout.strip().split(";")[0] + "\n")


def query_job_states(job_ids):
    """{job ID: SLURM state} from sacct; a job that sacct does not know yet is missing.

    A failed sacct call is retried, since sacct depends on the accounting database. After SACCT_ATTEMPTS failures in a
    row, all job_ids are cancelled with scancel and the call raises.
    """
    for attempt_number in range(1, SACCT_ATTEMPTS + 1):
        # with -j, sacct searches from Unix epoch 0 instead of only jobs since midnight
        sacct_result = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State",
                                      "-j", ",".join(job_ids)], capture_output=True, text=True)
        if sacct_result.returncode == 0:
            break
        print(f"WARNING: sacct failed (attempt {attempt_number} of {SACCT_ATTEMPTS}): {sacct_result.stderr.strip()}",
              file=sys.stderr, flush=True)
        time.sleep(SACCT_RETRY_INTERVAL_S)
    else:
        # without sacct no job can be followed any more; cancel them instead of leaving them to run unobserved
        scancel_result = subprocess.run(
            ["scancel", *job_ids], capture_output=True, text=True)
        raise RuntimeError(f"sacct seems to be broken: it did not respond for {SACCT_ATTEMPTS * SACCT_RETRY_INTERVAL_S // 60} "
                           f"minutes ({SACCT_ATTEMPTS} failed calls), last error: {sacct_result.stderr.strip()}. "
                           f"Cancelled jobs {', '.join(job_ids)}"
                           + (f"; scancel reported: {scancel_result.stderr.strip()}" if scancel_result.stderr.strip() else ""))
    state_by_job_id = {}
    for sacct_line in sacct_result.stdout.splitlines():
        job_id, state_text = sacct_line.split("|")
        # the state can carry extra text, e.g. "CANCELLED by <uid>"; keep only the state name
        state_by_job_id[job_id] = state_text.split()[0]
    return state_by_job_id


def poll_and_submit_jobs(stage_name, job_dirs, submitted_in_this_call):
    """One poll round: classify every job from outputs.json, its job_id and sacct, submit the jobs that need it, and add
    them to submitted_in_this_call. A job that had already failed before this call is resubmitted once.
    Returns (job_dirs still active, {failed job_dir: SLURM state}).
    """
    job_id_by_dir = {job_dir: (Path(job_dir) / "job_id").read_text().strip() for job_dir in job_dirs
                     if os.path.exists(os.path.join(job_dir, "job_id"))}
    state_by_job_id = query_job_states(
        list(job_id_by_dir.values())) if job_id_by_dir else {}
    done_dirs, active_dirs, failed_state_by_dir, dirs_to_submit = [], [], {}, []
    job_ids_not_in_sacct = []
    for job_dir in job_dirs:
        if os.path.exists(os.path.join(job_dir, "outputs.json")):
            done_dirs.append(job_dir)
        elif job_dir not in job_id_by_dir:
            dirs_to_submit.append(job_dir)
        # a job submitted moments ago can be missing from sacct; count it as pending
        elif state_by_job_id.get(job_id_by_dir[job_dir], "PENDING") in ACTIVE_STATES:
            active_dirs.append(job_dir)
            if job_id_by_dir[job_dir] not in state_by_job_id:
                job_ids_not_in_sacct.append(job_id_by_dir[job_dir])
        elif job_dir in submitted_in_this_call:
            failed_state_by_dir[job_dir] = state_by_job_id[job_id_by_dir[job_dir]]
        else:
            print(f"{stage_name}: resubmitting {job_dir} (previous job {job_id_by_dir[job_dir]} "
                  f"{state_by_job_id[job_id_by_dir[job_dir]]})", flush=True)
            dirs_to_submit.append(job_dir)
    for job_dir in dirs_to_submit:
        submit_job(job_dir)
        submitted_in_this_call.add(job_dir)
    # a job that stays unlisted for many polls is never counted as finished; the log shows which one it is
    not_in_sacct_note = (f", {len(job_ids_not_in_sacct)} not listed by sacct (job {', '.join(job_ids_not_in_sacct)})"
                         if job_ids_not_in_sacct else "")
    print(f"{stage_name}: {len(done_dirs)} done, {len(failed_state_by_dir)} failed, "
          f"{len(active_dirs) + len(dirs_to_submit)} active{not_in_sacct_note}", flush=True)
    return active_dirs + dirs_to_submit, failed_state_by_dir


def submit_and_wait_for_jobs(stage_name, job_dirs, poll_interval_s):
    """Submit every job not submitted yet and poll sacct until no job is pending or running.

    A job succeeded if it wrote job_dir/outputs.json. A job that fails during this call stays failed.
    Returns ({job_dir: job ID}, {failed job_dir: SLURM state}).
    """
    submitted_in_this_call = set()
    active_dirs, failed_state_by_dir = poll_and_submit_jobs(
        stage_name, job_dirs, submitted_in_this_call)
    while active_dirs:
        time.sleep(poll_interval_s)
        active_dirs, failed_state_by_dir = poll_and_submit_jobs(
            stage_name, job_dirs, submitted_in_this_call)
    return {job_dir: (Path(job_dir) / "job_id").read_text().strip() for job_dir in job_dirs}, failed_state_by_dir
