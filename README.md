# training_to_diffusion_pipeline

Workflow from MACE fine-tuning data to diffusion coefficients of clusters: cluster generation, MD, frame selection,
MLIP and VASP relaxation, VASP labelling, MACE fine-tuning, OPES enhanced sampling, diffusion MD and diffusion analysis.

Each stage is a module in `pipeline/`. The orchestrator chains stages from a TOML config and runs them locally or as
SLURM jobs.

## Installation

Requires Python 3.11 or newer.

```
pip install -e .
python -m pipeline.make_lookups
cp site_config.template.toml site_config.toml
```

`make_lookups` stores the MACE model names and VASP INCAR tags of the installed `mace-torch` and `ase` for the config
check. Rerun it after updating either package; the check fails until then.

Fill in `site_config.toml`:

- `job_setup`: shell lines run at the start of every SLURM job, e.g. module loads and environment activation.
- `[vasp]`: POTCAR directory and VASP executables.

VASP, POTCARs, PLUMED and SLURM are only needed for the stages that use them. Biased MD needs `PLUMED_KERNEL` to be set.

## Usage

Run a config from the directory its relative paths refer to:

```
python -m pipeline.orchestrator examples/chained/training_round.toml
```

`examples/chained/` contains full workflows, `examples/single_stage/` one config per stage; the two VASP stages share
`vasp.toml`. `examples/chained/production.toml` is the full pipeline with production settings. Replace every `<...>`
placeholder.

A single stage also runs without the orchestrator, in the current directory:

```
python -m pipeline.run_md examples/single_stage/run_md.toml start.traj
```

The module takes its `[[stage]]` from the config. Input files on the command line replace the stage's `inputs`.

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
module = "run_md"
name = "data_gen"
presets = ["mace"]
inputs = ["<path/to/start.traj>"]
nsteps = 2000
```

- `run_dir`: directory for all stage outputs.
- `[preset.<name>]`: shared settings. A stage includes them with `presets`; its own keys override them.
- `[[stage]]`: one stage. `module` is one of the stages below, `name` names its directory and output files.
- `inputs`: list of file paths and names of earlier stages, which can be mixed, e.g. `["analysis", "coordination"]` or
  `["diffusion", "extra.traj"]`. A stage name is replaced by the outputs of that stage. Without `inputs`, a stage takes
  the outputs of the previous stage. The config check fails if an entry is neither an earlier stage nor an existing
  file, or if it is both.
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
changed, its old directory is renamed `<dir>_old_<n>` and the stage and all later stages run again. If an item of a
stage fails, the pipeline stops after that stage; rerunning retries the failed items.

## Stages

### MD settings

Shared by `generate_cluster` and `run_md`.

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

### `generate_cluster`

Builds clusters by random placement in a sphere, BFGS relaxation and MD equilibration. One item per entry of
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

### `run_md`

Bussi MD, optionally biased with PLUMED. One item per input structure. Output: `<label>.traj`.

| Setting                  | Default  | Meaning                                                    |
| ------------------------ | -------- | ---------------------------------------------------------- |
| `nsteps`                 | required | MD steps                                                   |
| `plumed`                 | none     | PLUMED template, or a table `label = template` per input   |
| `start_structure_format` | none     | ASE format of the input structure                          |

In PLUMED templates, `@Pt@` is replaced by the indices of all Pt atoms. Use `UNITS LENGTH=A ENERGY=eV TIME=fs`.
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

### `finetune_mace`

Runs `mace.cli.run_train` on one `_train` and one `_val` input file. Output: `<mace_args.name>.model`.

| Setting     | Default  | Meaning                                                  |
| ----------- | -------- | -------------------------------------------------------- |
| `device`    | required | training device                                          |
| `mace_args` | required | `run_train` arguments, must include `name`; `true` gives a bare flag |

### `opes_minimum`

Picks the frame of an OPES trajectory closest to the free-energy minimum, computed from the `STATE` file next to the
trajectory. One item per input trajectory. Output: `<label>.traj`.

| Setting         | Default  | Meaning                          |
| --------------- | -------- | -------------------------------- |
| `temperature_K` | required | temperature of the OPES run      |
| `colvar_stride` | required | COLVAR rows per trajectory frame |
| `colvar_file`   | `COLVAR` | COLVAR file name                 |
| `state_file`    | `STATE`  | STATE file name                  |
| `fes_bins`      | none     | grid bins of the FES             |

### `analyze_diffusion`

Diffusion coefficient per species from blocked MSD fits. One item per input trajectory. Output:
`<label>_diffusion.csv` and plots.

| Setting            | Default                                        | Meaning                                                                                   |
| ------------------ | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `write_interval_fs`| required                                       | trajectory write interval                                                                 |
| `equilibration_ps` | required                                       | time discarded at the start                                                               |
| `cache_dir`        | required                                       | cache for intermediate results                                                            |
| `species.<El>`     | required                                       | `vacf_cutoff_ps`, `vacf_epsilon`, `n_samples_per_block`, `fit_start_ps`, `fit_width_ps`   |
| `analysis_stride`  | 1                                              | use every n-th frame                                                                      |
| `breakpoints_ps`   | `[]`                                           | times that split the trajectory; D is also computed per piece                             |
| `use_global_msd`   | false                                          | also fit the unblocked MSD                                                                |
| `rigid_body_check` | `{every_n_frames = 500, rel_tolerance = 1e-6}` | check of the rigid-body removal                                                           |

### `analyze_coordination`

Coordination numbers of every element pair, metal-metal pair distances and distances to the centre of mass. One item per
input trajectory. The result is kept in `cache_dir` and reused by later runs on the same trajectory with the same
settings. Output: the cached `coordination_<key>.npz`.

| Setting             | Default  | Meaning                                                 |
| ------------------- | -------- | ------------------------------------------------------- |
| `write_interval_fs` | required | trajectory write interval                               |
| `equilibration_ps`  | required | time discarded at the start                             |
| `cache_dir`         | required | cache for the results                                   |
| `stride`            | 10       | use every n-th frame                                    |
| `cutoff_factor`     | 1.1      | neighbour cutoff relative to the summed covalent radii  |

### `plot_diffusion`

Summary plots from the outputs of `analyze_diffusion` and `analyze_coordination`, e.g. `inputs = ["analysis",
"coordination"]`. System names must spell out the composition, e.g. `pt36c10` or `pt18ga18c5`. Output in the stage
directory:

- `overview/by_system/` and `overview/by_c_pt_ratio/`: one plot per group (`pt`, `ptga`) and one for all systems.
- `highlighted/<name>.png`: one plot per entry of `highlights`.
- `ratio/dc_over_dpt_<family>.png`: D_C / D_Pt per family.
- `coordination/<family>_d_<element>.png`: D and mean coordination numbers per family, and `cn_correlations.csv`.

| Setting              | Default | Meaning                                                                                   |
| -------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `regimes`            | `{}`    | named averages over `breakpoints_ps` pieces, e.g. `{pt36c1 = {surface = [1, 3], core = [2]}}` |
| `trend_regimes`      | `{}`    | regime replacing the global D of C in trend plots and of C and Pt in D_C / D_Pt, e.g. `{pt36c1 = "surface"}` |
| `highlights`         | `[]`    | `{name, group, systems, elements}` entries; `systems` may name regimes, e.g. `pt36c1_surface` |
| `coordination_files` | `{}`    | `{system = npz path}` for coordination files that are not stage inputs                    |

The `__main__` block of `pipeline/plot_diffusion.py` reproduces the report plots without the orchestrator; edit it to
change data paths, highlights or the plots made.

### `plot_structure`

Structure plots from the outputs of `analyze_coordination`, e.g. `inputs = ["coordination"]`. System names must spell out
the composition. Output in the stage directory:

- `pair_rdf_peak/<family>_<a>_<b>.png`: first pair-distance peak of every metal pair (Pt-Pt, Pt-Ga, Ga-Ga), split into
  pairs with and without a C-bonded atom. A population with fewer than 5 pairs per frame on average is left out.
- `metal_per_c/<family>.png`: distinct metal atoms with a C neighbour per C atom.

Both are plotted against C per metal atom: C / Pt for Pt clusters, C / (Pt+Ga) for PtGa clusters.
- `com_distance_time/<system>.png`: distance of every C atom from the centre of mass over time. Each colour follows one
  C atom; the colours carry no other meaning.
- `com_distance_density/<family>.png`: density of the distance from the centre of mass per element.

| Setting              | Default | Meaning                                                                |
| -------------------- | ------- | ---------------------------------------------------------------------- |
| `coordination_files` | `{}`    | `{system = npz path}` for coordination files that are not stage inputs |

The `__main__` block of `pipeline/plot_structure.py` makes these plots without the orchestrator.

## Acknowledgements

Developed with the help of Claude Opus 5.5.
