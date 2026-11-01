import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols

PLOT_STYLE = {
    # ACS figure guidelines: Arial/Helvetica, 8pt minimums
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{sfmath}",
    # ACS-compliant text sizes (8-10pt at final print size)
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
}
DPI = 600
ELEMENT_ORDER = ["C", "Pt", "Ga"]


def savefig_fsync(figure, path, **savefig_kwargs):
    """Save figure to path, close it and fsync the file."""
    figure.savefig(path, **savefig_kwargs)
    plt.close(figure)
    with open(path, "rb") as saved_file:
        os.fsync(saved_file.fileno())


def to_latex(text):
    """Wraps digit runs in a label in LaTeX subscript braces, e.g. Pt36 to Pt$_{36}$."""
    return re.sub(r"(\d+)", r"$_{\1}$", text)


def parse_composition(system):
    """Element counts from a system name such as pt18ga18c5, in name order: {"Pt": 18, "Ga": 18, "C": 5}."""
    element_counts = re.findall(r"([a-z]+)(\d+)", system)
    composition = {element.capitalize(): int(count)
                   for element, count in element_counts}
    if "".join(element + count for element, count in element_counts) != system or not set(composition) <= set(chemical_symbols):
        raise ValueError(
            f"system name {system!r} does not spell out a composition such as pt36c10 or pt18ga18c5")
    return composition


def display_label(category):
    """LaTeX label of a system or regime category, e.g. pt36c1_surface to Pt$_{36}$C$_{1}$ (Surface)."""
    system, _, regime = category.partition("_")
    formula = "".join(f"{element}{count}" for element,
                      count in parse_composition(system).items())
    return to_latex(formula) + (f" ({regime.capitalize()})" if regime else "")


def family_key(category):
    """Maps a system key to its family, e.g. pt36c10 / pt36c1_surface to pt36, pt18ga18c15 to pt18ga18."""
    return re.sub(r"c\d+$", "", str(category).partition("_")[0])


def c_pt_ratio(system):
    """Number of C atoms per Pt atom of a system."""
    composition = parse_composition(system)
    return composition.get("C", 0) / composition["Pt"]


def c_metal_ratio(system):
    """Number of C atoms per metal atom of a system, e.g. C/(Pt+Ga)."""
    composition = parse_composition(system)
    return composition.get("C", 0) / sum(count for element, count in composition.items() if element != "C")


def c_metal_label(system):
    """Axis label of c_metal_ratio, e.g. "C / Pt" or "C / (Pt+Ga)"."""
    metals = [element for element in parse_composition(
        system) if element != "C"]
    return "C / " + (metals[0] if len(metals) == 1 else f"({'+'.join(metals)})")


def system_sort_key(system):
    """Sorts Pt before PtGa systems, then by metal counts, then by C count."""
    composition = parse_composition(system)
    metal_counts = [count for element,
                    count in composition.items() if element != "C"]
    return len(metal_counts), metal_counts, composition.get("C", 0)


def systems_by_family(systems):
    """{family: systems sorted by C count}, families in sort order."""
    families = {}
    for system in sorted(set(systems), key=system_sort_key):
        families.setdefault(family_key(system), []).append(system)
    return families


def carbon_systems(systems):
    """The systems that contain carbon."""
    return [system for system in systems if parse_composition(system).get("C", 0) > 0]


def coordination_files_by_system(npz_paths, coordination_files):
    """{system: npz path} of analyze_coordination files: the system_name stored in each file, then coordination_files on top."""
    npz_path_by_system = {}
    for npz_path in npz_paths:
        with np.load(npz_path) as coordination:
            npz_path_by_system[str(coordination["system_name"])] = npz_path
    return {**npz_path_by_system, **coordination_files}


def ordered_elements(elements):
    """Elements in ELEMENT_ORDER first, then any others alphabetically."""
    elements = set(elements)
    return [element for element in ELEMENT_ORDER if element in elements] + sorted(elements - set(ELEMENT_ORDER))


def element_color(element):
    """Matplotlib color of an element, the same in every plot: C0 for C, C1 for Pt, C2 for Ga."""
    return f"C{ELEMENT_ORDER.index(element)}" if element in ELEMENT_ORDER else "black"


def save_plot(figure, path):
    """Save figure to path, creating its directory, and return path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    savefig_fsync(figure, path, dpi=DPI, bbox_inches="tight")
    return path
