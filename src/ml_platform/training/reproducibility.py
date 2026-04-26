from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducible experiments."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_rng_state() -> dict[str, Any]:
    return {"python_random_state": random.getstate()}
