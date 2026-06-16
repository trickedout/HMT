from __future__ import annotations

import os
import random


def set_seed(seed: int = 3407) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
