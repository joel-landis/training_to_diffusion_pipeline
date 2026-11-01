"""Write lookups.json for the installed mace-torch and ase; rerun after updating either package."""
from importlib.metadata import version

from ase.calculators.vasp import create_input
from mace.calculators.foundations_models import mace_mp_names

from pipeline.checks import LOOKUPS_PATH
from pipeline.io_utils import write_json

if __name__ == "__main__":
    incar_tags = sorted({tag.upper() for tag_list in (create_input.float_keys, create_input.exp_keys, create_input.string_keys,
                                                      create_input.int_keys, create_input.list_int_keys, create_input.list_bool_keys,
                                                      create_input.list_float_keys, create_input.special_keys, create_input.dict_keys,
                                                      create_input.bool_keys)
                         for tag in tag_list})
    write_json(LOOKUPS_PATH, {
        # mace_mp_names starts with None, the default model; None is not a name
        "mace-torch": {"version": version("mace-torch"), "mace_mp_names": [name for name in mace_mp_names if name is not None]},
        "ase": {"version": version("ase"), "incar_tags": incar_tags},
    })
    print(f"wrote {LOOKUPS_PATH}", flush=True)
