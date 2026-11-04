import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

HERE = Path(__file__).parent


def run_python(args, cwd, env=None):
    result = subprocess.run([sys.executable, *map(str, args)],
                            cwd=cwd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-5000:]
    return result


def copy_config(name, tmp_path):
    shutil.copy(HERE / name, tmp_path)
    return name


def read_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def copy_structures(tmp_path):
    for path in (HERE / "fixtures" / "structures").iterdir():
        shutil.copy(path, tmp_path)


def test_generate_cluster(tmp_path):
    shutil.copy(HERE / "single" / "gen.toml", tmp_path)
    run_python(["-m", "pipeline.orchestrator", "gen.toml"], tmp_path)
    unit_dir = tmp_path / "run" / "00_cluster" / "cluster"
    for name in ("cluster.traj", "cluster_relaxed.traj", "cluster_equilibration.traj", "outputs.json"):
        assert (unit_dir / name).exists(), name


def test_run_md(tmp_path):
    shutil.copy(HERE / "single" / "md.toml", tmp_path)
    copy_structures(tmp_path)
    run_python(["-m", "pipeline.orchestrator", "md.toml"], tmp_path)
    unit_dir = tmp_path / "run" / "00_md" / "pt36c1_start"
    for name in ("pt36c1_start.traj", "pt36c1_start.log", "outputs.json"):
        assert (unit_dir / name).exists(), name


def test_select_frames(tmp_path):
    shutil.copy(HERE / "single" / "select.toml", tmp_path)
    copy_structures(tmp_path)
    run_python(["-m", "pipeline.orchestrator", "select.toml"], tmp_path)
    stage_dir = tmp_path / "run" / "00_selection"
    for name in ("selection_train.traj", "selection_val.traj", "selection_provenance.csv", "outputs.json"):
        assert (stage_dir / name).exists(), name


def test_relax_mlip(tmp_path):
    shutil.copy(HERE / "single" / "relax_mlip.toml", tmp_path)
    write(tmp_path / "frames_train.traj", read(HERE / "fixtures" /
          "structures" / "pt36c1_diffusion_21frames.traj", index=":2"))
    run_python(["-m", "pipeline.orchestrator", "relax_mlip.toml"], tmp_path)
    stage_dir = tmp_path / "run" / "00_relax"
    for name in ("relax_train.traj", "relax_relax.csv", "outputs.json"):
        assert (stage_dir / name).exists(), name


def test_several_structures(tmp_path):
    copy_structures(tmp_path)
    stdout = run_python(["-m", "pipeline.orchestrator",
                        copy_config("test_multi.toml", tmp_path)], tmp_path).stdout
    for label in ("pt36c1_start", "pt18ga18c5_start"):
        assert (tmp_path / "run_multi" / "00_md" /
                label / f"{label}.traj").exists()
    assert len(read_json(tmp_path / "run_multi" /
               "00_md" / "outputs.json")) == 2
    assert "finished:" in stdout


def test_skip_done_stages(tmp_path):
    copy_structures(tmp_path)
    run_python(["-m", "pipeline.orchestrator",
               copy_config("test_stage_0.toml", tmp_path)], tmp_path)
    run_dir = tmp_path / "run_skip"
    traj = run_dir / "00_equilibration" / "pt36c1_start" / "pt36c1_start.traj"
    assert (run_dir / "00_equilibration" / "outputs.json").exists()
    assert not (run_dir / "01_production").exists()
    mtime = os.path.getmtime(traj)

    stdout = run_python(["-m", "pipeline.orchestrator",
                        copy_config("test_stages_0_1.toml", tmp_path)], tmp_path).stdout
    assert "[00_equilibration] md already done" in stdout
    assert os.path.getmtime(traj) == mtime
    assert (run_dir / "01_production" /
            "pt36c1_start" / "outputs.json").exists()


def test_rerun_changed_settings(tmp_path):
    copy_structures(tmp_path)
    run_python(["-m", "pipeline.orchestrator",
               copy_config("test_stage_0.toml", tmp_path)], tmp_path)
    run_dir = tmp_path / "run_skip"
    mtime = os.path.getmtime(
        run_dir / "00_equilibration" / "pt36c1_start" / "pt36c1_start.traj")

    stdout = run_python(["-m", "pipeline.orchestrator",
                        copy_config("test_stage_0_changed.toml", tmp_path)], tmp_path).stdout
    assert "[00_equilibration] md settings or inputs changed" in stdout
    assert os.path.getmtime(run_dir / "00_equilibration_old_1" /
                            "pt36c1_start" / "pt36c1_start.traj") == mtime
    assert (run_dir / "00_equilibration" / "outputs.json").exists()

    stdout = run_python(["-m", "pipeline.orchestrator",
                        "test_stage_0_changed.toml"], tmp_path).stdout
    assert "[00_equilibration] md already done" in stdout
    assert not (run_dir / "00_equilibration_old_2").exists()


def test_failed_work_unit_stops_pipeline(tmp_path):
    orchestrator_command = [
        sys.executable, "-m", "pipeline.orchestrator", copy_config("test_fail.toml", tmp_path)]
    cluster = tmp_path / "run_fail" / "00_cluster"
    result = subprocess.run(orchestrator_command,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "unplaceable failed: RuntimeError: Failed to place" in result.stderr
    assert "cluster: 1 of 2 work units failed" in result.stderr
    assert not (cluster / "outputs.json").exists()
    assert (cluster / "pt36c1" / "outputs.json").exists()
    mtime = os.path.getmtime(cluster / "pt36c1" / "pt36c1.traj")

    result = subprocess.run(orchestrator_command,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "[00_cluster] generate_clusters resuming in" in result.stdout
    assert os.path.getmtime(cluster / "pt36c1" / "pt36c1.traj") == mtime
    assert "unplaceable failed" in result.stderr


def test_slurm_resubmits_failed_once(tmp_path, monkeypatch):
    from pipeline import slurm_jobs
    dirs = {name: tmp_path /
            name for name in ("done", "failed_before", "unsubmitted")}
    for job_dir in dirs.values():
        job_dir.mkdir()
    (dirs["done"] / "outputs.json").write_text("[]")
    (dirs["done"] / "job_id").write_text("1\n")
    (dirs["failed_before"] / "job_id").write_text("2\n")
    states = {"1": "COMPLETED", "2": "FAILED"}
    submitted = []

    def fake_submit(job_dir):
        job_id = str(len(states) + 1)
        states[job_id] = "FAILED"
        submitted.append(os.path.basename(job_dir))
        (Path(job_dir) / "job_id").write_text(job_id + "\n")

    monkeypatch.setattr(slurm_jobs, "submit_job", fake_submit)
    monkeypatch.setattr(slurm_jobs, "query_job_states", lambda job_ids: {
                        job_id: states[job_id] for job_id in job_ids})
    _, failed = slurm_jobs.submit_and_wait_for_jobs(
        "test", [str(job_dir) for job_dir in dirs.values()], 0)
    assert sorted(submitted) == ["failed_before", "unsubmitted"]
    assert set(failed) == {
        str(dirs["failed_before"]), str(dirs["unsubmitted"])}


def test_presets():
    from pipeline import config
    pipeline_config = {"preset": {"a": {"model": "m", "incar": {
        "ENCUT": 400, "EDIFF": 1e-5}}, "b": {"device": "cuda"}}}
    stage = {"task": "label_vasp", "presets": [
        "a", "b"], "incar": {"ENCUT": 300, "NSW": 3}}
    assert config.merge_stage_settings(pipeline_config, stage) == {
        "task": "label_vasp", "model": "m", "device": "cuda", "incar": {"ENCUT": 300, "EDIFF": 1e-5, "NSW": 3}}
    assert pipeline_config["preset"]["a"]["incar"] == {
        "ENCUT": 400, "EDIFF": 1e-5}


def test_slurm_gpu_request():
    from pipeline import slurm_jobs
    md_settings = {"name": "production",
                   "slurm": {"time": "72:00:00", "gpus": 1}}
    job_script = slurm_jobs.slurm_job_script(
        md_settings, "run/01_production/pt36c1", "production_pt36c1")
    assert "#SBATCH --gpus=1\n" in job_script and "#SBATCH --time=72:00:00" in job_script
    job_script = slurm_jobs.slurm_job_script(
        {**md_settings, "slurm": {"gpus": "rtx_4090:1"}}, "run/01_production/pt36c1", "production_pt36c1")
    assert "#SBATCH --gpus=rtx_4090:1\n" in job_script
    job_script = slurm_jobs.slurm_job_script(
        {"name": "selection", "slurm": {}}, "run/02_selection", "selection")
    assert "#SBATCH --time=24:00:00" in job_script and "--gpus" not in job_script


def test_ncore_for():
    from pipeline import vasp_jobs
    assert [vasp_jobs.ncore_for(ntasks) for ntasks in (
        1, 4, 10, 15, 20, 120, 7, 13)] == [1, 1, 2, 3, 5, 30, 2, 4]


def test_finetune_mace(tmp_path):
    frames = read(HERE / "fixtures" / "structures" /
                  "pt36c1_diffusion_21frames.traj", index=":6")
    for frame_index, atoms in enumerate(frames):
        atoms.info["dft_energy"] = -180.0 - 0.1 * frame_index
        atoms.arrays["dft_forces"] = np.full(
            (len(atoms), 3), 0.01 * frame_index)
        atoms.calc = None
    write(tmp_path / "data_train.extxyz", frames[:4], format="extxyz")
    write(tmp_path / "data_val.extxyz", frames[4:], format="extxyz")
    run_python(["-m", "pipeline.orchestrator",
               copy_config("test_finetune.toml", tmp_path)], tmp_path)
    finetune = tmp_path / "run_finetune" / "00_finetune"
    for name in ("tiny.model", "outputs.json"):
        assert (finetune / name).exists(), name


def test_plumed_input(tmp_path):
    from pipeline import ase_simulation
    from ase import Atoms
    template = tmp_path / "plumed.dat"
    template.write_text(
        "UNITS LENGTH=A ENERGY=eV TIME=fs\n# @Ga@ in a comment\n\nc: COORDINATION GROUPA=@Pt@ GROUPB=@C@ R_0=2.7\n")
    atoms = Atoms("PtCPt", positions=np.zeros((3, 3)))
    assert ase_simulation.plumed_input_lines(template, atoms) == [
        "UNITS LENGTH=A ENERGY=eV TIME=fs", "c: COORDINATION GROUPA=1,3 GROUPB=2 R_0=2.7"]
    template.write_text("c: COORDINATION GROUPA=@Ga@ GROUPB=@C@ R_0=2.7\n")
    with pytest.raises(ValueError, match="placeholder @Ga@"):
        ase_simulation.plumed_input_lines(template, atoms)
    from pipeline.stage_specs import WORK_UNITS
    work_units = WORK_UNITS["md"]({"name": "opes", "plumed": {
                               "pt36c1": "ptc.dat", "pt18ga18c5": "ptgac.dat"}}, ["a/pt36c1.traj", "b/pt18ga18c5.traj"])
    assert [(label, unit["plumed"]) for label, unit, _ in work_units] == [
        ("pt36c1", "ptc.dat"), ("pt18ga18c5", "ptgac.dat")]


def test_opes_minimum(tmp_path):
    for path in (HERE / "fixtures" / "opes").iterdir():
        shutil.copy(path, tmp_path)
    shutil.copy(HERE / "single" / "opes_minimum.toml", tmp_path)
    run_python(["-m", "pipeline.orchestrator", "opes_minimum.toml"], tmp_path)
    unit_dir = tmp_path / "run" / "00_opes_minimum" / "opes"
    for name in ("opes.traj", "opes_fes.dat", "outputs.json"):
        assert (unit_dir / name).exists(), name
