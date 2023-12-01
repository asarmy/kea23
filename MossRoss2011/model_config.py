import sys
from pathlib import Path

# Add path for project
# FIXME: shouldn't need to do this!
PROJ_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_DIR))
del PROJ_DIR

# Import project configurations
import project_config  # noqa: F401

# Recommended magnitude / allowable styles
MAG_RANGE = {"reverse": (5.5, 8.0)}
