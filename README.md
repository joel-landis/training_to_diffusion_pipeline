# training_to_diffusion_pipeline

Workflow from MACE fine-tuning data to diffusion coefficients of clusters: cluster generation, MD, frame selection,
MLIP and VASP relaxation, VASP labelling, MACE fine-tuning, OPES enhanced sampling, diffusion MD and diffusion analysis.

Each task is a function in `pipeline/stages.py`, built from the MACE/ASE, VASP, analysis and plot modules in `pipeline/`.
The orchestrator chains stages from a TOML config and runs them locally or as SLURM jobs.

## Installation

Requires Python 3.11 or newer.

```
pip install -e .
cp site_config.template.toml site_config.toml
```

Fill in `site_config.toml`:

- `job_setup`: shell lines run at the start of every SLURM job, e.g. module loads and environment activation.
- `[vasp]`: POTCAR directory and VASP executables.

VASP, POTCARs, PLUMED and SLURM are only needed for the stages that use them. Biased MD needs `PLUMED_KERNEL` to be set.
The `plot` stage renders text with LaTeX and needs a LaTeX installation with the `helvet` and `sfmath` packages.

## Usage

Run a config from the directory its relative paths refer to:

```
python -m pipeline.orchestrator examples/chained/training_round.toml
```

`examples/chained/` contains full workflows, `examples/single_stage/` one config per task; the two VASP tasks share
`vasp.toml`. `examples/chained/production.toml` runs from cluster generation to the diffusion MD with
production settings. Replace every `<...>`
placeholder.

A single stage runs with a config that holds only that stage:

```
python -m pipeline.orchestrator examples/single_stage/md.toml
```

A Python script can also call the task functions directly. They take the output directory, the input (one path, a list
of paths, or none for `generate_clusters`) and the stage settings as keyword arguments, and return the output paths. The
output directory must exist:

```python
from pipeline import stages

stages.md("md_run", "start.traj", name="md", model="mh-1", device="cuda", temperature_K=873.0, timestep_fs=1.0,
          taut_fs=20.0, nsteps=2000, write_interval_fs=10.0, log_interval_fs=10.0)
```

## Changes from v0.1

The previous version is on the branch `v.0.1`. The pipeline does the same; configs written for v0.1 need these
adjustments:

- The stage key `module` is now `task`. Renamed and merged tasks, with the example config that replaces the old one:

  | v0.1 `module`                               | `task`                                         | Example config                                                                       |
  | ------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
  | `generate_cluster`                          | `generate_clusters`                            | [`generate_clusters.toml`](examples/single_stage/generate_clusters.toml)             |
  | `run_md`                                    | `md`                                           | [`md.toml`](examples/single_stage/md.toml)                                           |
  | `finetune_mace`                             | `finetune`                                     | [`finetune.toml`](examples/single_stage/finetune.toml)                               |
  | `analyze_diffusion`, `analyze_coordination` | `analyze`, one stage with the settings of both | [`analyze.toml`](examples/single_stage/analyze.toml)                                 |
  | `plot_diffusion`, `plot_structure`          | `plot`, one stage with the settings of both    | [`plot.toml`](examples/single_stage/plot.toml)                                       |

  `select_frames`, `relax_mlip`, `relax_vasp`, `label_vasp` and `opes_minimum` are unchanged.
- GPUs: `device = "cuda"` no longer requests a GPU by itself, and `slurm.gpu_class` is gone. Request GPUs with
  `slurm.gpus`, e.g. `gpus = 1` or `gpus = "a100:1"`.
- A setting that needs the output of an earlier stage names that stage, e.g. `model = "finetune"`, instead of a path
  into the run directory. Every file path in the config must exist when the pipeline starts.
- `analyze` takes only trajectories. The plots are a separate `plot` stage, e.g. `inputs = ["analysis"]`.
- A single stage no longer runs as `python -m pipeline.<module> config.toml inputs`. Use a config with only that stage,
  or call the task function from Python (see Usage).
- Stage directories are numbered from `01_` instead of `00_`. Run directories of v0.1 are not picked up: a rerun
  starts with the first stage.

## Config

```toml
run_dir = "run"

[preset.mace]
model = "mh-1"
head = "matpes_r2scan"
device = "cuda"
default_dtype = "float32"
slurm = { gpus = 1 }

[[stage]]
task = "md"
name = "data_gen"
presets = ["mace"]
inputs = ["<path/to/start.traj>"]
nsteps = 2000
```

- `run_dir`: directory for all stage outputs.
- `[preset.<name>]`: shared settings. A stage includes them with `presets`; they are merged in the listed order, and the
  stage's own keys are merged last. Tables such as `slurm` and `incar` are merged key by key: a stage's
  `slurm = { time = "04:00:00" }` keeps the `gpus` of its preset.
- `[[stage]]`: one stage. `task` is one of the tasks below, `name` names its directory and output files.
- `inputs`: list of file paths and names of earlier stages, which can be mixed, e.g. `["data_gen_opes", "data_gen_md"]`
  or `["analysis", "old_results_diffusion.csv"]`. A stage name is replaced by the outputs of that stage. Without `inputs`, a stage takes
  the outputs of the previous stage. The config check fails if an entry is neither an earlier stage nor an existing
  file, or if it is both.
- A setting whose value is the name of an earlier stage gets that stage's output, e.g. `model = "finetune"` for the model
  of a `finetune` stage. The same check applies.
- `model`, `head`, `device`, `default_dtype`, `dispersion`, `damping`, `dispersion_xc`: MACE calculator, passed to
  `mace_mp`.
- `slurm`: run the stage as SLURM jobs. Without it, the stage runs in the orchestrator process.

| Key               | Default    | Meaning                                 |
| ----------------- | ---------- | --------------------------------------- |
| `time`            | `24:00:00` | `--time`                                |
| `cpus_per_task`   | 1          | `--cpus-per-task`                       |
| `mem_per_cpu`     | `6G`       | `--mem-per-cpu`                         |
| `gpus`            | none       | `--gpus`, e.g. `1` or `"a100:1"`        |
| `sbatch_args`     | `[]`       | extra `#SBATCH` lines                   |
| `poll_interval_s` | 60         | seconds between job status checks       |

## Restarting

Rerunning the same config skips finished stages and resumes unfinished ones. If the settings or inputs of a stage
changed, its old directory is renamed `<dir>_old_<n>` and the stage and all later stages run again. If a work unit of
a stage fails, the pipeline stops after that stage; rerunning retries the failed work units.

## Stages

### MD settings

Shared by `generate_clusters` and `md`.

| Setting                         | Default  | Meaning                                                      |
| ------------------------------- | -------- | ------------------------------------------------------------ |
| `temperature_K`                 | required | thermostat temperature                                       |
| `timestep_fs`                   | required | MD timestep                                                  |
| `taut_fs`                       | required | Bussi thermostat time constant                               |
| `write_interval_fs`             | required | trajectory write interval                                    |
| `log_interval_fs`               | required | log interval                                                 |
| `seed`                          | none     | random seed                                                  |
| `remove_rigid_body_interval_fs` | 10.0     | interval for removing centre-of-mass translation and rotation |

The thermostat targets the internal temperature of the cluster, so the temperature in `<name>.log` reads lower than
`temperature_K`.

### `generate_clusters`

Builds clusters by random placement in a sphere, BFGS relaxation and MD equilibration. One work unit per entry of
`clusters`. Output: `<label>.traj`.

| Setting                | Default  | Meaning                                                                   |
| ---------------------- | -------- | ------------------------------------------------------------------------- |
| `clusters`             | required | `label = {composition = {Pt = 36, C = 1}}`; other keys override settings |
| `radius`               | required | placement sphere radius in Å                                              |
| `equilibration_nsteps` | required | MD steps                                                                  |
| `fmax`                 | 0.1      | BFGS force threshold in eV/Å                                              |
| `opt_max_steps`        | 200      | maximum BFGS steps                                                        |
| `vacuum`               | 10.0     | vacuum around the cluster in Å                                            |
| `c_c_separation`       | 3.5      | minimum C–C distance at placement in Å                                    |
| `cov_scale`            | 0.8      | minimum distance of other pairs relative to the summed covalent radii    |
| `max_attempts`         | 50000    | placement attempts per atom                                               |

### `md`

Bussi MD, optionally biased with PLUMED. One work unit per input structure. Output: `<label>.traj`.

| Setting                  | Default  | Meaning                                                    |
| ------------------------ | -------- | ---------------------------------------------------------- |
| `nsteps`                 | required | MD steps                                                   |
| `plumed`                 | none     | PLUMED template, or a table `label = template` per input   |
| `start_structure_format` | none     | ASE format of the input structure                          |

In PLUMED templates, `@<element>@` is replaced by the indices of all atoms of that element, e.g. `@Pt@`. Use `UNITS LENGTH=A ENERGY=eV TIME=fs`.
Examples are in `examples/plumed/`.

### `select_frames`

Selects frames of each trajectory by farthest point sampling on MACE descriptors and splits them into train and val.
Output: `<name>_train.traj`, `<name>_val.traj`.

| Setting                  | Default  | Meaning                                                  |
| ------------------------ | -------- | -------------------------------------------------------- |
| `frames_per_traj`        | required | frames selected per trajectory                           |
| `device`                 | required | device for MACE and the sampling                         |
| `thin_to`                | 3000     | frames per trajectory considered                         |
| `atoms_per_frame_factor` | 5        | atoms sampled per selected frame                         |
| `val_fraction`           | 0.2      | fraction of frames in val                                |
| `seed`                   | 42       | seed of the split                                        |
| `exclude_provenance`     | none     | `_provenance.csv` of an earlier selection to exclude     |

### `relax_mlip`

Relaxes every frame whose largest force exceeds `fmax` and appends the relaxed copy. Inputs are split files such as
`selection_train.traj`. Output: one `<name>_<split>.traj` per input.

| Setting         | Default | Meaning                  |
| --------------- | ------- | ------------------------ |
| `fmax`          | 0.1     | force threshold in eV/Å  |
| `opt_max_steps` | 500     | maximum BFGS steps       |

### `relax_vasp`, `label_vasp`

Both submit one VASP job per frame to SLURM and need no `slurm` key. Inputs are split files.

`relax_vasp` relaxes the frames relaxed by `relax_mlip`, or adds relaxed copies of all frames if there are none.
Output: `<name>_<split>.traj`.

`label_vasp` computes single points. Output: `<name>_<split>.xyz` with `dft_energy` and `dft_forces`.

| Setting           | Default  | Meaning                                                   |
| ----------------- | -------- | --------------------------------------------------------- |
| `incar`           | required | INCAR tags                                                |
| `potcars`         | required | POTCAR subdirectory per element, e.g. `{Ga = "Ga_d"}`     |
| `ntasks`          | required | MPI ranks                                                 |
| `time`            | required | walltime                                                  |
| `mem_per_cpu`     | `2G`     | `--mem-per-cpu`                                           |
| `kpoints`         | none     | `{mesh = [a, b, c], gamma = true}`; default uses `KSPACING` |
| `sbatch_args`     | `[]`     | extra `#SBATCH` lines                                     |
| `sync_interval_s` | 60       | interval for copying results back from node scratch      |
| `poll_interval_s` | 60       | seconds between job status checks                         |

`NCORE` is set from `ntasks` unless given in `incar`. Failed frames are listed with a reason in `<name>_status.csv`.
Rerunning the config resubmits them; a relaxation that reached `NSW` continues from its last geometry.

### `finetune`

Runs `mace.cli.run_train` on one `_train` and one `_val` input file. Output: `<mace_args.name>.model`.

| Setting     | Default  | Meaning                                                  |
| ----------- | -------- | -------------------------------------------------------- |
| `device`    | required | training device                                          |
| `mace_args` | required | `run_train` arguments, must include `name`; `true` gives a bare flag |

### `opes_minimum`

Picks the frame of an OPES trajectory closest to the free-energy minimum, computed from the `STATE` file next to the
trajectory. One work unit per input trajectory. Output: `<label>.traj`.

| Setting         | Default  | Meaning                          |
| --------------- | -------- | -------------------------------- |
| `temperature_K` | required | temperature of the OPES run      |
| `colvar_stride` | required | COLVAR rows per trajectory frame |
| `colvar_file`   | `COLVAR` | COLVAR file name                 |
| `state_file`    | `STATE`  | STATE file name                  |
| `fes_bins`      | none     | grid bins of the FES             |

### `analyze`

Diffusion coefficient per species from blocked MSD fits, and coordination numbers of every element pair, metal-metal pair
distances and distances to the centre of mass. One work unit per input trajectory. Give the stage a `slurm` key, e.g.
`slurm = { mem_per_cpu = "2G" }`; each trajectory then runs as its own job. Output: `<label>_diffusion.csv`, the VACF and
MSD plots, and the coordination result `coordination_<key>.npz` in `cache_dir`, which later runs on the same trajectory
with the same settings reuse.

| Setting             | Default                                        | Meaning                                                                                 |
| ------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `write_interval_fs` | required                                       | trajectory write interval                                                               |
| `equilibration_ps`  | required                                       | time discarded at the start                                                             |
| `cache_dir`         | required                                       | cache for intermediate results                                                          |
| `species.<El>`      | required                                       | `vacf_cutoff_ps`, `vacf_epsilon`, `n_samples_per_block`, `fit_start_ps`, `fit_width_ps` |
| `analysis_stride`   | 1                                              | use every n-th frame for the diffusion analysis                                         |
| `breakpoints_ps`    | `[]`                                           | times that split the trajectory; D is also computed per piece                           |
| `use_global_msd`    | false                                          | also fit the unblocked MSD                                                              |
| `rigid_body_check`  | `{every_n_frames = 500, rel_tolerance = 1e-6}` | check of the rigid-body removal                                                         |
| `stride`            | 10                                             | use every n-th frame for the coordination analysis                                      |
| `cutoff_factor`     | 1.1                                            | neighbour cutoff relative to the summed covalent radii                                  |

### `plot`

Summary and structure plots from `<label>_diffusion.csv` and coordination npz files, e.g. `inputs = ["analysis"]`, also
together with result files of earlier runs. System names must spell out the composition, e.g. `pt36c10` or `pt18ga18c5`.
Output in the stage directory:

- `overview/by_system/` and `overview/by_c_pt_ratio/`: one plot per group (`pt`, `ptga`) and one for all systems.
- `highlighted/<name>.png`: one plot per entry of `highlights`.
- `ratio/dc_over_dpt_<family>.png`: D_C / D_Pt per family.
- `coordination/<family>_d_<element>.png`: D and mean coordination numbers per family, and `cn_correlations.csv`.
- `pair_rdf_peak/<family>_<a>_<b>.png`: first pair-distance peak of every metal pair (Pt-Pt, Pt-Ga, Ga-Ga), split into
  pairs with and without a C-bonded atom. A population with fewer than 5 pairs per frame on average is left out.
- `metal_per_c/<family>.png`: distinct metal atoms with a C neighbour per C atom.
- `com_distance_time/<system>.png`: distance of every C atom from the centre of mass over time. Each colour follows one
  C atom; the colours carry no other meaning.
- `com_distance_density/<family>.png`: density of the distance from the centre of mass per element.

`pair_rdf_peak` and `metal_per_c` are plotted against C per metal atom: C / Pt for Pt clusters, C / (Pt+Ga) for PtGa
clusters.

| Setting              | Default | Meaning                                                                                   |
| -------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `regimes`            | `{}`    | named averages over `breakpoints_ps` pieces, e.g. `{pt36c1 = {surface = [1, 3], core = [2]}}` |
| `trend_regimes`      | `{}`    | regime replacing the global D of C in trend plots and of C and Pt in D_C / D_Pt, e.g. `{pt36c1 = "surface"}` |
| `highlights`         | `[]`    | `{name, group, systems, elements}` entries; `systems` may name regimes, e.g. `pt36c1_surface` |
| `coordination_files` | `{}`    | `{system = npz path}` for coordination files that are not stage inputs                    |

The `__main__` blocks of `pipeline/plot_diffusion.py` and `pipeline/plot_structure.py` make the report plots without the
orchestrator; edit them to change data paths, highlights or the plots made.

## Acknowledgements

Developed with the help of Claude Opus 5.5.
