# qgear/toolbox/__init__.py

import os
import importlib

# Dynamically import all Python modules in this folder
_pkg_dir = os.path.dirname(__file__)
for _fname in os.listdir(_pkg_dir):
    if _fname.endswith(".py") and _fname != "__init__.py":
        _mod_name = f"{__name__}.{_fname[:-3]}"
        globals()[_fname[:-3]] = importlib.import_module(_mod_name)