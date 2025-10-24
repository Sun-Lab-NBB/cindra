from typing import Any
from pathlib import Path
from dataclasses import field, asdict, fields, dataclass

import numpy as np
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..version import version, sl_version, python_version

# base_config.py
from dataclasses import dataclass, field, asdict
from typing import Any, Literal
from pathlib import Path
import numpy as np


@dataclass
class S2PConfigBase:
    """Configuration superclass for suite2p SingleDay and MultiDay configs.
    """
    ops_source: Literal["config", "newly_added"] = "config"
    mode: Literal["tiff", "binary", "raw"] = "tiff"

