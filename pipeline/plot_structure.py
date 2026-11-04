import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.plot_utils import (PLOT_STYLE, c_metal_label, c_metal_ratio, carbon_systems, display_label,
                                 element_color, ordered_elements, parse_composition, save_plot, systems_by_family)

PEAK_BIN_WIDTH = 0.003
PEAK_RANGE = (1.0, 4.0)
# a population with fewer Pt-Pt pairs per frame on average gives no reliable peak and is left out of the plot
MIN_PAIR_COUNT = 5
ROLLING_WINDOW_PS = 10.0
COM_BIN_WIDTH = 0.1


def pair_distance_peaks(npz_path, first_element, second_element):
    """First pair-distance peak in Å of first-second pairs with at least one C-bonded atom ("c_bonded") and of the other pairs ("none").

    An atom is C-bonded in a frame if a C atom is within the coordination cutoff. The peak is the tallest PEAK_BIN_WIDTH bin
    in PEAK_RANGE; a population with fewer than MIN_PAIR_COUNT pairs per frame on average gets None.
    """
    with np.load(npz_path) as coordination:
        pair_distances = coordination[f"distances_{first_element}_{second_element}"]
        c_bonded = {}
        for element in {first_element, second_element}:
            if f"counts_C_around_{element}" in coordination.files:
                c_bonded[element] = coordination[f"counts_C_around_{element}"] > 0
            else:
                c_bonded[element] = np.zeros(
                    (len(pair_distances), int(coordination[f"n_{element}"])), dtype=bool)
    if first_element == second_element:
        first_atoms, second_atoms = np.triu_indices(
            c_bonded[first_element].shape[1], 1)
        pair_has_c_bonded_end = c_bonded[first_element][:,
                                                        first_atoms] | c_bonded[first_element][:, second_atoms]
    else:
        pair_has_c_bonded_end = (c_bonded[first_element][:, :, None] |
                                 c_bonded[second_element][:, None, :]).reshape(len(pair_distances), -1)
    bin_edges = np.arange(0.0, PEAK_RANGE[1] + PEAK_BIN_WIDTH, PEAK_BIN_WIDTH)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    in_peak_range = bin_centres >= PEAK_RANGE[0]
    peaks = {}
    for population, in_population in (("c_bonded", pair_has_c_bonded_end), ("none", ~pair_has_c_bonded_end)):
        population_distances = pair_distances[in_population]
        if population_distances.size / len(pair_distances) < MIN_PAIR_COUNT:
            peaks[population] = None
            continue
        bin_counts, _ = np.histogram(population_distances, bins=bin_edges)
        peaks[population] = bin_centres[in_peak_range][bin_counts[in_peak_range].argmax()]
    return peaks


def plot_pair_distance_peaks(npz_path_by_system, output_dir):
    """pair_rdf_peak/<family>_<a>_<b>.png: first a-b peak of pairs with and without a C-bonded atom against C per metal atom,
    for every metal pair of every family with carbon."""
    plot_paths = []
    for family, systems in systems_by_family(npz_path_by_system).items():
        if not carbon_systems(systems):
            continue
        metals = sorted(element for element in parse_composition(
            systems[0]) if element != "C")
        for first_element, second_element in itertools.combinations_with_replacement(metals, 2):
            peaks_by_system = {system: pair_distance_peaks(
                npz_path_by_system[system], first_element, second_element) for system in systems}
            pair_name = first_element if first_element == second_element else "-".join(
                ordered_elements([first_element, second_element])) + " pair"
            # bond length next to carbon and away from it, against carbon loading
            figure, ax = plt.subplots(figsize=(7, 3.5))
            for population, color, label in (("c_bonded", "tab:orange", f"C bonded {pair_name}"),
                                             ("none", "tab:blue", f"Non C bonded {pair_name}")):
                points = [(c_metal_ratio(system), peaks_by_system[system][population]) for system in systems
                          if peaks_by_system[system][population] is not None]
                if points:
                    ax.scatter(*zip(*points), color=color,
                               edgecolors="black", label=label)
            for system in carbon_systems(systems):
                if None not in peaks_by_system[system].values():
                    print(f"{system} {first_element}-{second_element}: {c_metal_label(system)} {c_metal_ratio(system):.3f}, peak C bonded - none = "
                          f"{peaks_by_system[system]['c_bonded'] - peaks_by_system[system]['none']:+.3f} A", flush=True)
            ax.set_xlabel(c_metal_label(systems[0]))
            ax.set_ylabel(r"peak r [\AA]")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            figure.tight_layout()
            plot_paths.append(save_plot(figure, os.path.join(output_dir, "pair_rdf_peak",
                                                             f"{family}_{first_element.lower()}_{second_element.lower()}.png")))
    return plot_paths


def metal_atoms_per_c(npz_path):
    """{metal element: number of distinct atoms of it with a C neighbour, per C atom}, averaged over frames."""
    with np.load(npz_path) as coordination:
        n_c = int(coordination["n_C"])
        return {str(element): float((coordination[f"counts_C_around_{element}"] > 0).sum(axis=1).mean() / n_c)
                for element in coordination["unique_elements"] if element != "C"}


def plot_metal_atoms_per_c(npz_path_by_system, output_dir):
    """metal_per_c/<family>.png: distinct metal atoms with a C neighbour per C atom against C per metal atom, per family with carbon."""
    plot_paths = []
    for family, systems in systems_by_family(npz_path_by_system).items():
        loaded_systems = carbon_systems(systems)
        if not loaded_systems:
            continue
        values_by_system = {system: metal_atoms_per_c(
            npz_path_by_system[system]) for system in loaded_systems}
        # a metal atom bonded to several C atoms counts once, so a decrease means that C atoms share metal atoms
        figure, ax = plt.subplots(figsize=(7, 3.5))
        for element in ordered_elements(values_by_system[loaded_systems[0]]):
            ax.scatter([c_metal_ratio(system) for system in loaded_systems], [values_by_system[system][element] for system in loaded_systems],
                       color=element_color(element), edgecolors="black", label=element)
        ax.set_xlabel(c_metal_label(loaded_systems[0]))
        ax.set_ylabel("Average number of metal atoms per C")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        figure.tight_layout()
        plot_paths.append(save_plot(figure, os.path.join(
            output_dir, "metal_per_c", f"{family}.png")))
    return plot_paths


def plot_com_distance_traces(npz_path_by_system, output_dir, frame_interval_ps=None, rolling_window_ps=ROLLING_WINDOW_PS):
    """com_distance_time/<system>.png: distance of every C atom from the centre of mass over time, raw and as rolling mean.

    frame_interval_ps: time between stored frames; None reads frame_interval_fs from the npz.
    """
    plot_paths = []
    for system in carbon_systems(sorted(npz_path_by_system)):
        with np.load(npz_path_by_system[system]) as coordination:
            carbon_distances = coordination["dist_to_com_C"]
            system_frame_interval_ps = float(
                coordination["frame_interval_fs"]) / 1000.0 if frame_interval_ps is None else frame_interval_ps
        time_ps = np.arange(len(carbon_distances)) * system_frame_interval_ps
        rolling_distances = pd.DataFrame(carbon_distances).rolling(window=max(1, round(rolling_window_ps / system_frame_interval_ps)),
                                                                   min_periods=1, center=True).mean().to_numpy()
        # when each C atom sits at the surface or in the core of the cluster
        figure, ax = plt.subplots(figsize=(7, 3.5))
        for carbon_index in range(carbon_distances.shape[1]):
            ax.plot(time_ps, carbon_distances[:, carbon_index],
                    color=f"C{carbon_index % 10}", alpha=0.18, linewidth=1.2)
            ax.plot(time_ps, rolling_distances[:, carbon_index],
                    color=f"C{carbon_index % 10}", linewidth=3.0)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"Carbon distance from COM [\AA]")
        ax.set_ylim(0, None)
        figure.tight_layout()
        plot_paths.append(save_plot(figure, os.path.join(
            output_dir, "com_distance_time", f"{system}.png")))
    return plot_paths


def plot_com_distance_densities(npz_path_by_system, output_dir):
    """com_distance_density/<family>.png: density of the distance from the centre of mass per element, one panel per system."""
    plot_paths = []
    for family, systems in systems_by_family(npz_path_by_system).items():
        distances_by_system = {}
        for system in systems:
            with np.load(npz_path_by_system[system]) as coordination:
                distances_by_system[system] = {element: coordination[f"dist_to_com_{element}"].reshape(-1)
                                               for element in parse_composition(system)}
        bin_edges = np.arange(0.0, math.ceil(max(values.max() for distances in distances_by_system.values()
                                                 for values in distances.values())) + COM_BIN_WIDTH, COM_BIN_WIDTH)
        n_columns = 2 if len(systems) == 4 else min(len(systems), 3)
        n_rows = math.ceil(len(systems) / n_columns)
        # where each element sits inside the cluster: surface or core
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(
            7, 1.75 * n_rows), sharex=True, sharey=True, squeeze=False)
        for ax, system in zip(axes.ravel(), systems):
            for element in ordered_elements(distances_by_system[system]):
                density, _ = np.histogram(
                    distances_by_system[system][element], bins=bin_edges, density=True)
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]),
                        density, color=element_color(element), label=element)
            ax.set_title(display_label(system))
        for unused_ax in axes.ravel()[len(systems):]:
            unused_ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"distance from COM [\AA]")
        for ax in axes[:, 0]:
            ax.set_ylabel("density")
        axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        figure.tight_layout()
        plot_paths.append(save_plot(figure, os.path.join(
            output_dir, "com_distance_density", f"{family}.png")))
    return plot_paths


if __name__ == "__main__":
    # Standalone use without the orchestrator: structure plots from analyze_coordination npz files into output_dir.
    # Edit the paths and calls below; comment out a call to skip that group of plots.
    output_dir = "structure_plots"
    npz_path_by_system = {
        "pt36": "<path/to/pt36_coordination.npz>",
        "pt36c1": "<path/to/pt36c1_coordination.npz>",
        "pt36c5": "<path/to/pt36c5_coordination.npz>",
        "pt36c10": "<path/to/pt36c10_coordination.npz>",
        "pt36c20": "<path/to/pt36c20_coordination.npz>",
        "pt36c30": "<path/to/pt36c30_coordination.npz>",
        "pt18ga18": "<path/to/pt18ga18_coordination.npz>",
        "pt18ga18c1": "<path/to/pt18ga18c1_coordination.npz>",
        "pt18ga18c3": "<path/to/pt18ga18c3_coordination.npz>",
        "pt18ga18c5": "<path/to/pt18ga18c5_coordination.npz>",
        "pt18ga18c10": "<path/to/pt18ga18c10_coordination.npz>",
        "pt18ga18c15": "<path/to/pt18ga18c15_coordination.npz>",
    }

    with plt.rc_context(PLOT_STYLE):
        # pair_rdf_peak/<family>_<a>_<b>.png
        plot_pair_distance_peaks(npz_path_by_system, output_dir)
        # metal_per_c/<family>.png
        plot_metal_atoms_per_c(npz_path_by_system, output_dir)
        # com_distance_time/<system>.png; for npz files without frame_interval_fs pass frame_interval_ps, e.g. 0.1
        plot_com_distance_traces(npz_path_by_system, output_dir)
        # com_distance_density/<family>.png
        plot_com_distance_densities(npz_path_by_system, output_dir)
