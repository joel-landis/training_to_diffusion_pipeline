import itertools
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.plot_utils import (PLOT_STYLE, c_pt_ratio, carbon_systems, display_label, family_key,
                                 ordered_elements, parse_composition, save_plot, system_sort_key, systems_by_family)
from pipeline.io_utils import write_csv

Y_LABEL = r"D [m$^{2}$/s]"
RESULTS_COLUMNS = ["system", "regime", "d", "sem", "element"]
# (centre, neighbour) pairs of the coordination panels: mean number of neighbour atoms around each centre atom
CN_PAIRS = [("C", "C"), ("Pt", "C"), ("Pt", "Pt"),
            ("Ga", "Ga"), ("Pt", "Ga"), ("Ga", "C")]


def group_name(system):
    """Metal elements of a system in name order, e.g. pt18ga18c5 to ptga."""
    return "".join(element.lower() for element in parse_composition(system) if element != "C")


def systems_by_group(systems):
    """{group: sorted systems} for every group, e.g. pt and ptga, plus "all" with every system."""
    sorted_systems = sorted(set(systems), key=system_sort_key)
    groups = {}
    for system in sorted_systems:
        groups.setdefault(group_name(system), []).append(system)
    groups["all"] = sorted_systems
    return groups


def family_spaced_positions(category_order, group_gap):
    """Bar x positions with group_gap extra spacing inserted wherever the family key changes."""
    x = np.arange(len(category_order), dtype=float)
    if not group_gap:
        return x
    family_keys = [family_key(category) for category in category_order]
    shift = 0.0
    for position_index in range(1, len(family_keys)):
        if family_keys[position_index] != family_keys[position_index - 1]:
            shift += group_gap
        x[position_index] += shift
    return x


def fold_yaxis_offset_into_label(ax, base_label, **label_kwargs):
    """Moves matplotlib's scientific-notation y-axis offset text into the axis label itself."""
    ax.figure.canvas.draw()
    offset_text = ax.yaxis.get_offset_text().get_text()
    ax.yaxis.offsetText.set_visible(False)
    label = f"{base_label} $\\small\\text{{({offset_text})}}$" if offset_text else base_label
    ax.set_ylabel(label, **label_kwargs)


def build_grouped_tables(results, category_col, category_order, hue_order):
    """Pivots long-form (category, element, d, sem) rows into per-category-per-element value and SEM tables for grouped bar plotting."""
    sorted_results = results.copy()
    sorted_results[category_col] = pd.Categorical(
        sorted_results[category_col], categories=category_order, ordered=True)
    sorted_results["element"] = pd.Categorical(
        sorted_results["element"], categories=hue_order, ordered=True)
    sorted_results = sorted_results.sort_values([category_col, "element"])

    values = sorted_results.pivot(
        index=category_col, columns="element", values="d").reindex(category_order)
    sems = sorted_results.pivot(
        index=category_col, columns="element", values="sem").reindex(category_order)
    return sorted_results, values, sems


def plot_grouped_bars(
    values,
    sems,
    category_order,
    display_labels,
    hue_order,
    *,
    console_prefix,
    xlabel,
    ylabel,
    figsize=(7, 3.5),
    ax=None,
    connect_means=False,
    linewidth=2.0,
    alpha_map=None,
    element_highlight=None,
    group_gap=0.6,
):
    """Grouped bar chart of D with SEM error bars per category and element, optionally onto an existing axis.

    Categories are spaced by group_gap wherever the family changes. Elements not in element_highlight are faded.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = family_spaced_positions(category_order, group_gap)
    width = 0.8 / max(len(hue_order), 1)
    offsets = (np.arange(len(hue_order), dtype=float) -
               (len(hue_order) - 1) / 2.0) * width
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    if len(colors) < len(hue_order):
        colors = [f"C{color_index}" for color_index in range(len(hue_order))]

    print(console_prefix)
    for offset, color, element in zip(offsets, colors, hue_order):
        if element in values.columns:
            heights = values[element]
            errors = sems[element]
        else:
            heights = pd.Series(np.nan, index=category_order, dtype=float)
            errors = pd.Series(np.nan, index=category_order, dtype=float)

        has_value = heights.notna()
        categories = heights[has_value].index.tolist()
        bar_positions = x[has_value.to_numpy()] + offset
        bar_heights = heights[has_value].to_numpy(dtype=float)
        bar_errors = errors[has_value].fillna(0.0).to_numpy(dtype=float)

        if alpha_map is None:
            base_alphas = [1.0] * len(categories)
        else:
            base_alphas = [alpha_map.get(category, 1.0)
                           for category in categories]

        if element_highlight is None or element in element_highlight:
            alphas = base_alphas
        else:
            alphas = [0.2 if alpha == 1.0 else alpha for alpha in base_alphas]

        first_label = True
        for alpha_value in sorted(set(alphas), reverse=True):
            bar_indices = [bar_index for bar_index,
                           alpha in enumerate(alphas) if alpha == alpha_value]
            ax.bar(
                bar_positions[bar_indices],
                bar_heights[bar_indices],
                width=width,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=alpha_value,
                label=element if first_label else None,
            )
            ax.errorbar(
                bar_positions[bar_indices],
                bar_heights[bar_indices],
                yerr=bar_errors[bar_indices],
                fmt="none",
                ecolor="black",
                elinewidth=2.0,
                capsize=6,
                alpha=alpha_value,
            )
            first_label = False

        if connect_means:
            ax.plot(bar_positions, bar_heights, color=color,
                    linewidth=linewidth, linestyle="-", zorder=3)

        for category, d_value, sem_value in zip(categories, bar_heights, bar_errors):
            print(
                f"{xlabel}={category} | element={element}: {d_value:.6g} +/- {sem_value:.6g}")

    ax.set_xticks(x, display_labels)
    ax.set_ylabel(ylabel)
    fold_yaxis_offset_into_label(ax, ylabel)
    ax.tick_params(axis="both")
    legend = ax.legend(title="element", loc="upper left",
                       bbox_to_anchor=(1.02, 1))
    for legend_handle in legend.legend_handles:
        legend_handle.set_alpha(1.0)
    return fig, ax


def load_diffusion_results(results_paths):
    """D and SEM in m^2/s per system, regime and element, from <label>_diffusion.csv files of analyze_diffusion or from
    tables that already have the columns system, regime, d, sem, element (D in Å^2/ps).

    The span all of analyze_diffusion becomes regime global, piece_<n> becomes regime <n>.
    """
    results_tables = []
    for results_path in results_paths:
        results_table = pd.read_csv(results_path)
        if "D_mean_A2_per_ps" in results_table.columns:
            results_table = pd.DataFrame({
                "system": os.path.basename(results_path).removesuffix("_diffusion.csv"),
                "regime": results_table["span"].replace({"all": "global"}).str.removeprefix("piece_"),
                "d": results_table["D_mean_A2_per_ps"],
                "sem": results_table["D_sem_A2_per_ps"],
                "element": results_table["species"],
            })
        missing_columns = set(RESULTS_COLUMNS) - set(results_table.columns)
        if missing_columns:
            raise ValueError(f"{results_path}: neither a <label>_diffusion.csv nor a table with {RESULTS_COLUMNS}; "
                             f"missing {sorted(missing_columns)}")
        results_tables.append(results_table[RESULTS_COLUMNS])
    diffusion_results = pd.concat(results_tables, ignore_index=True)
    diffusion_results["regime"] = diffusion_results["regime"].astype(str)
    # convert D, sem from A^2/ps to SI units (m^2/s): 1 A^2/ps = 1e-8 m^2/s
    diffusion_results["d"] = diffusion_results["d"] * 1e-8
    diffusion_results["sem"] = diffusion_results["sem"] * 1e-8
    return diffusion_results


def load_coordination_means(npz_path_by_system):
    """Mean coordination number of every CN_PAIRS pair per system, from analyze_coordination npz files.

    npz_path_by_system: {system: npz path}. Pairs with an element the system does not contain are 0.
    Columns: system and "<centre>-<neighbour> CN".
    """
    coordination_rows = []
    for system, npz_path in npz_path_by_system.items():
        composition = parse_composition(system)
        coordination_row = {"system": system}
        with np.load(npz_path) as coordination:
            for element, count in composition.items():
                if int(coordination[f"n_{element}"]) != count:
                    raise ValueError(
                        f"{npz_path} has {int(coordination[f'n_{element}'])} {element} atoms, but system {system} has {count}")
            for centre, neighbour in CN_PAIRS:
                has_pair = centre in composition and neighbour in composition
                coordination_row[f"{centre}-{neighbour} CN"] = float(
                    coordination[f"cn_{neighbour}_{centre}"].mean()) if has_pair else 0.0
        coordination_rows.append(coordination_row)
    return pd.DataFrame(coordination_rows)


def regime_results(diffusion_results, regimes):
    """D and SEM of every named regime: mean over its pieces, SEM = sqrt(sum of squared piece SEMs) / number of pieces.

    regimes: {system: {regime name: [piece numbers]}}.
    """
    regime_rows = []
    for system, pieces_by_regime in regimes.items():
        for regime, piece_numbers in pieces_by_regime.items():
            pieces = diffusion_results[(diffusion_results["system"] == system)
                                       & diffusion_results["regime"].isin([str(piece_number) for piece_number in piece_numbers])]
            if pieces.empty:
                raise ValueError(
                    f"no pieces {piece_numbers} of {system} for regime {regime}")
            for element, element_pieces in pieces.groupby("element"):
                if len(element_pieces) != len(piece_numbers):
                    raise ValueError(
                        f"{system} {element}: regime {regime} needs pieces {piece_numbers}, found {element_pieces['regime'].tolist()}")
                regime_rows.append({"system": system, "regime": regime, "element": element, "d": element_pieces["d"].mean(),
                                    "sem": np.sqrt((element_pieces["sem"] ** 2).sum()) / len(piece_numbers)})
    return pd.DataFrame(regime_rows, columns=["system", "regime", "element", "d", "sem"])


def bar_results(diffusion_results, regimes):
    """Global D of every system plus the named regimes, with bar category <system> or <system>_<regime>."""
    global_results = diffusion_results[diffusion_results["regime"] == "global"]
    named_regime_results = regime_results(diffusion_results, regimes)
    return pd.concat([global_results.assign(category=global_results["system"]),
                      named_regime_results.assign(category=named_regime_results["system"] + "_" + named_regime_results["regime"])],
                     ignore_index=True)[["category", "system", "element", "d", "sem"]]


def category_order(systems, regimes):
    """Bar order: systems in sort order, each preceded by its named regimes."""
    return [category for system in sorted(systems, key=system_sort_key)
            for category in [f"{system}_{regime}" for regime in regimes.get(system, {})] + [system]]


def trend_results(diffusion_results, regimes, trend_regimes, regime_elements):
    """Global D per system and element, with the value of regime trend_regimes[system] for the elements in regime_elements."""
    trend_table = diffusion_results[diffusion_results["regime"] == "global"][[
        "system", "element", "d", "sem"]].copy()
    named_regime_results = regime_results(diffusion_results, regimes)
    for system, regime in trend_regimes.items():
        for element in regime_elements:
            replacement = named_regime_results[(named_regime_results["system"] == system) & (named_regime_results["regime"] == regime)
                                               & (named_regime_results["element"] == element)]
            if len(replacement) != 1:
                raise ValueError(
                    f"trend_regimes: no regime {regime!r} of {system} for {element}; define it in regimes")
            trend_rows = (trend_table["system"] == system) & (
                trend_table["element"] == element)
            trend_table.loc[trend_rows, ["d", "sem"]
                            ] = replacement[["d", "sem"]].to_numpy()
    return trend_table


def plot_bar_chart(bars, systems, regimes, path, alpha_map=None, element_highlight=None):
    """Grouped bar chart of the given systems and their named regimes, saved to path."""
    order = category_order(systems, regimes)
    selected_bars = bars[bars["category"].isin(order)]
    elements = ordered_elements(selected_bars["element"])
    _, values, sems = build_grouped_tables(
        selected_bars[["category", "element", "d", "sem"]], "category", order, elements)
    figure, ax = plt.subplots(figsize=(max(7.0, 0.25 * len(order)), 3.5))
    plot_grouped_bars(values, sems, order, [display_label(category) for category in order], elements,
                      console_prefix=f"{path} (value +/- sem):", xlabel="System", ylabel=Y_LABEL, ax=ax,
                      alpha_map=alpha_map, element_highlight=element_highlight)
    ax.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return save_plot(figure, path)


def plot_c_pt_ratio_chart(trends, systems, path):
    """D against C/Pt, one panel per element and one line per family, saved to path."""
    selected_trends = trends[trends["system"].isin(systems)]
    families = systems_by_family(systems)
    family_colors = {family: f"C{family_index}" for family_index,
                     family in enumerate(families)}
    elements = ordered_elements(selected_trends["element"])
    figure, axes = plt.subplots(
        1, len(elements), figsize=(7, 2.5), squeeze=False)
    for ax, element in zip(axes[0], elements):
        for family, family_systems in families.items():
            family_trends = selected_trends[selected_trends["system"].isin(
                family_systems) & (selected_trends["element"] == element)]
            family_trends = family_trends.assign(
                c_pt_ratio=family_trends["system"].map(c_pt_ratio)).sort_values("c_pt_ratio")
            if not family_trends.empty:
                ax.errorbar(family_trends["c_pt_ratio"], family_trends["d"], yerr=family_trends["sem"], fmt="o-", capsize=3,
                            color=family_colors[family], label=display_label(family))
        ax.set_title(element)
        ax.set_xlabel("C / Pt ratio")
        fold_yaxis_offset_into_label(ax, Y_LABEL)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.tight_layout()
    return save_plot(figure, path)


def plot_overviews(diffusion_results, regimes, trend_regimes, output_dir):
    """Overview plots of every group (pt, ptga, ...) and of all systems together.

    overview/by_system/<group>.png: bars per system and element, named regimes next to their system.
    overview/by_c_pt_ratio/<group>.png: D against C/Pt per family; C uses the trend_regimes values.
    """
    bars = bar_results(diffusion_results, regimes)
    trends = trend_results(diffusion_results, regimes, trend_regimes, ["C"])
    plot_paths = []
    for group, systems in systems_by_group(diffusion_results["system"]).items():
        plot_paths.append(plot_bar_chart(bars, systems, regimes, os.path.join(
            output_dir, "overview", "by_system", f"{group}.png")))
        # D against carbon loading; families of different size share one axis
        plot_paths.append(plot_c_pt_ratio_chart(trends, systems, os.path.join(
            output_dir, "overview", "by_c_pt_ratio", f"{group}.png")))
    return plot_paths


def plot_highlights(diffusion_results, regimes, highlights, output_dir):
    """highlighted/<name>.png per highlights entry: the bar chart of its group with its systems and elements opaque, the rest faded.

    highlights: [{"name": ..., "group": "pt", "ptga", ... or "all", "systems": [categories, e.g. "pt36c1_surface"], "elements": [...]}].
    """
    bars = bar_results(diffusion_results, regimes)
    groups = systems_by_group(diffusion_results["system"])
    plot_paths = []
    for highlight in highlights:
        if highlight["group"] not in groups:
            raise ValueError(
                f"highlight {highlight['name']}: group {highlight['group']!r} not in {sorted(groups)}")
        order = category_order(groups[highlight["group"]], regimes)
        unknown_categories = set(highlight["systems"]) - set(order)
        if unknown_categories:
            raise ValueError(
                f"highlight {highlight['name']}: {sorted(unknown_categories)} are not bars of group {highlight['group']}")
        # one family and element emphasised, all others faded for context
        alpha_map = {
            category: 1.0 if category in highlight["systems"] else 0.2 for category in order}
        plot_paths.append(plot_bar_chart(bars, groups[highlight["group"]], regimes,
                                         os.path.join(
                                             output_dir, "highlighted", f"{highlight['name']}.png"),
                                         alpha_map=alpha_map, element_highlight=highlight["elements"]))
    return plot_paths


def plot_diffusion_ratios(diffusion_results, regimes, trend_regimes, output_dir):
    """ratio/dc_over_dpt_<family>.png: D_C / D_Pt against C/Pt for every family with carbon.

    Both D of a system in trend_regimes come from that regime; the error is propagated from both SEMs.
    """
    trends = trend_results(diffusion_results, regimes,
                           trend_regimes, ["C", "Pt"])
    carbon_trends = trends[trends["element"] == "C"].set_index("system")
    platinum_trends = trends[trends["element"] == "Pt"].set_index("system")
    plot_paths = []
    for family, family_systems in systems_by_family(carbon_trends.index.intersection(platinum_trends.index)).items():
        ratio = carbon_trends.loc[family_systems, "d"] / \
            platinum_trends.loc[family_systems, "d"]
        ratio_error = ratio * np.sqrt((carbon_trends.loc[family_systems, "sem"] / carbon_trends.loc[family_systems, "d"]) ** 2
                                      + (platinum_trends.loc[family_systems, "sem"] / platinum_trends.loc[family_systems, "d"]) ** 2)
        # relative mobility of C and Pt; the dashed line marks equal mobility
        figure, ax = plt.subplots(figsize=(7, 3.5))
        ax.errorbar([c_pt_ratio(system) for system in family_systems], ratio, yerr=ratio_error,
                    fmt="o", color="C0", markeredgecolor="black", capsize=3)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel("C / Pt ratio")
        ax.set_ylabel(r"$D_C / D_{Pt}$")
        ax.set_ylim(0, None)
        figure.tight_layout()
        plot_paths.append(save_plot(figure, os.path.join(
            output_dir, "ratio", f"dc_over_dpt_{family}.png")))
    return plot_paths


def family_cn_columns(family_systems):
    """Coordination columns of the CN_PAIRS whose elements occur in the family."""
    family_elements = {
        element for system in family_systems for element in parse_composition(system)}
    return [f"{centre}-{neighbour} CN" for centre, neighbour in CN_PAIRS if {centre, neighbour} <= family_elements]


def plot_coordination_twin_chart(element_trends, coordination_means, cn_columns, path):
    """D (left axis) and mean coordination numbers (right axis) against C/Pt, one panel per coordination column, saved to path."""
    merged = element_trends.merge(coordination_means, on="system")
    merged = merged.assign(c_pt_ratio=merged["system"].map(
        c_pt_ratio)).sort_values("c_pt_ratio")
    n_rows = math.ceil(len(cn_columns) / 3)
    figure, axes = plt.subplots(n_rows, 3, figsize=(
        7, 2.1875 if n_rows == 1 else 1.75 * n_rows), squeeze=False)
    for ax, cn_column in zip(axes.ravel(), cn_columns):
        twin_ax = ax.twinx()
        ax.errorbar(merged["c_pt_ratio"], merged["d"],
                    yerr=merged["sem"], fmt="o-", color="tab:blue", capsize=3)
        twin_ax.plot(merged["c_pt_ratio"], merged[cn_column],
                     "s--", color="tab:orange")
        ax.set_xlabel("C / Pt ratio")
        fold_yaxis_offset_into_label(ax, Y_LABEL, color="tab:blue")
        twin_ax.set_ylabel(cn_column, color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        twin_ax.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title(cn_column)
    for unused_ax in axes.ravel()[len(cn_columns):]:
        unused_ax.set_visible(False)
    figure.tight_layout()
    return save_plot(figure, path)


def plot_coordination_trends(diffusion_results, coordination_means, regimes, trend_regimes, output_dir):
    """coordination/<family>_d_<element>.png for every family with coordination data and every element of it.

    C uses the trend_regimes values, Pt and Ga the global ones. If the family has a system in trend_regimes, Pt gets a second
    plot <family>_d_pt_<regime>.png with the trend_regimes values.
    """
    global_trends = trend_results(diffusion_results, regimes, {}, [])
    trends_by_regime_element = {element: trend_results(
        diffusion_results, regimes, trend_regimes, [element]) for element in ["C", "Pt"]}
    plot_paths = []
    for family, family_systems in systems_by_family(coordination_means["system"]).items():
        family_coordination = coordination_means[coordination_means["system"].isin(
            family_systems)]
        cn_columns = family_cn_columns(family_systems)
        family_elements = ordered_elements(
            {element for system in family_systems for element in parse_composition(system)})
        family_trend_regimes = sorted(
            {regime for system, regime in trend_regimes.items() if system in family_systems})
        variants = []
        for element in family_elements:
            if element == "C":
                variants.append(
                    (element, trends_by_regime_element["C"], f"{family}_d_c.png"))
                continue
            variants.append(
                (element, global_trends, f"{family}_d_{element.lower()}.png"))
            if element == "Pt" and family_trend_regimes:
                variants.append(
                    (element, trends_by_regime_element["Pt"], f"{family}_d_pt_{'_'.join(family_trend_regimes)}.png"))
        for element, trends, file_name in variants:
            element_trends = trends[(
                trends["element"] == element) & trends["system"].isin(family_systems)]
            # D (left axis) and mean coordination number (right axis) against C/Pt, to compare trends
            plot_paths.append(plot_coordination_twin_chart(element_trends, family_coordination, cn_columns,
                                                           os.path.join(output_dir, "coordination", file_name)))
    return plot_paths


def write_coordination_correlations(coordination_means, output_dir):
    """coordination/cn_correlations.csv: Pearson r between every two coordination columns, over the systems with carbon of each family."""
    correlation_rows = []
    for family, family_systems in systems_by_family(coordination_means["system"]).items():
        loaded_systems = carbon_systems(family_systems)
        family_coordination = coordination_means.set_index(
            "system").loc[loaded_systems]
        for first_column, second_column in itertools.combinations(family_cn_columns(family_systems), 2):
            pearson_r = np.corrcoef(
                family_coordination[first_column], family_coordination[second_column])[0, 1]
            correlation_rows.append(
                [family, first_column, second_column, len(loaded_systems), pearson_r])
    correlations_path = os.path.join(
        output_dir, "coordination", "cn_correlations.csv")
    os.makedirs(os.path.dirname(correlations_path), exist_ok=True)
    write_csv(correlations_path, [
              "family", "cn_a", "cn_b", "n_systems", "pearson_r"], correlation_rows)
    return correlations_path


if __name__ == "__main__":
    # Standalone use without the orchestrator: reproduces the plots of the diffusion report into output_dir.
    # Edit the paths, settings and calls below; comment out a call to skip that group of plots.
    output_dir = "diffusion_plots"
    diffusion_results = load_diffusion_results(["<path/to/results.csv>"])
    coordination_means = load_coordination_means({
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
    })

    # Pt36C1: C is mobile on the surface (pieces 1 and 3) and immobile in the core (piece 2); C trends use the surface value
    regimes = {"pt36c1": {"surface": [1, 3], "core": [2]}}
    trend_regimes = {"pt36c1": "surface"}

    pt36_family = ["pt36", "pt36c1_surface", "pt36c1_core",
                   "pt36c1", "pt36c5", "pt36c10", "pt36c20", "pt36c30"]
    pt72_pt144_families = ["pt72", "pt72c20", "pt72c40", "pt144",
                           "pt144c40", "pt144c60", "pt36", "pt36c10", "pt36c20"]
    pt18ga18_family = ["pt18ga18", "pt18ga18c1", "pt18ga18c3",
                       "pt18ga18c5", "pt18ga18c10", "pt18ga18c15"]
    pt36ga36_pt72ga72_families = ["pt36ga36", "pt36ga36c10", "pt36ga36c20", "pt36ga36c30", "pt72ga72", "pt72ga72c20", "pt72ga72c40",
                                  "pt72ga72c60", "pt18ga18", "pt18ga18c5", "pt18ga18c10", "pt18ga18c15"]
    pt96ga48_family = ["pt96ga48", "pt96ga48c25", "pt96ga48c50", "pt96ga48c75"]
    highlights = [
        {"name": "fig01_pt36_pt", "group": "pt",
            "systems": pt36_family, "elements": ["Pt"]},
        {"name": "fig02_pt36_c", "group": "pt",
            "systems": pt36_family, "elements": ["C"]},
        {"name": "fig09_pt72_pt144_pt", "group": "pt",
            "systems": pt72_pt144_families, "elements": ["Pt"]},
        {"name": "fig10_pt72_pt144_c", "group": "pt",
            "systems": pt72_pt144_families, "elements": ["C"]},
        {"name": "fig11_pt18ga18_pt", "group": "ptga",
            "systems": pt18ga18_family, "elements": ["Pt"]},
        {"name": "fig12_pt18ga18_c", "group": "ptga",
            "systems": pt18ga18_family, "elements": ["C"]},
        {"name": "fig13_pt18ga18_ga", "group": "ptga",
            "systems": pt18ga18_family, "elements": ["Ga"]},
        {"name": "fig18_pt36ga36_pt72ga72_pt", "group": "ptga",
            "systems": pt36ga36_pt72ga72_families, "elements": ["Pt"]},
        {"name": "fig19_pt36ga36_pt72ga72_c", "group": "ptga",
            "systems": pt36ga36_pt72ga72_families, "elements": ["C"]},
        {"name": "fig20_pt36ga36_pt72ga72_ga", "group": "ptga",
            "systems": pt36ga36_pt72ga72_families, "elements": ["Ga"]},
        {"name": "fig21_pt96ga48", "group": "ptga",
            "systems": pt96ga48_family, "elements": ["C", "Pt", "Ga"]},
        # own highlight: any bars of a group ("pt", "ptga" or "all") and any elements, e.g. Pt of the bare clusters across all systems
        {"name": "bare_clusters_pt", "group": "all", "systems": ["pt36", "pt72", "pt144", "pt18ga18", "pt36ga36", "pt72ga72", "pt96ga48"],
         "elements": ["Pt"]},
    ]

    with plt.rc_context(PLOT_STYLE):
        # overview/by_system/ and overview/by_c_pt_ratio/: pt, ptga and all systems
        plot_overviews(diffusion_results, regimes, trend_regimes, output_dir)
        # highlighted/<name>.png for every entry of highlights
        plot_highlights(diffusion_results, regimes, highlights, output_dir)
        # ratio/dc_over_dpt_<family>.png
        plot_diffusion_ratios(diffusion_results, regimes,
                              trend_regimes, output_dir)
        # coordination/<family>_d_<element>.png for the families in coordination_means
        plot_coordination_trends(
            diffusion_results, coordination_means, regimes, trend_regimes, output_dir)
        # coordination/cn_correlations.csv
        write_coordination_correlations(coordination_means, output_dir)
