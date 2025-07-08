"""
Source code for hypercube projection paper.

Authors: Yoshiaki Horiike & Shin Fujishiro
"""

__version__ = "1.0.0"
__author__ = "Yoshiaki Horiike & Shin Fujishiro"

# Import modules when available
import importlib.util

__all__ = []

# Check for module availability and import if present
if importlib.util.find_spec(".hypercube", package=__name__):
    from . import hypercube as hypercube  # Redundant alias to satisfy linter

    __all__.append("hypercube")

if importlib.util.find_spec(".networks", package=__name__):
    from . import networks as networks  # Redundant alias to satisfy linter

    __all__.append("networks")

if importlib.util.find_spec(".dynamics", package=__name__):
    from . import dynamics as dynamics  # Redundant alias to satisfy linter

    __all__.append("dynamics")
