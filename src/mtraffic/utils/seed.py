"""Global determinism helpers."""

from __future__ import annotations

import os
import random


def set_all(seed: int) -> None:
    """Seed every random source we know about.

    Frameworks are imported lazily so callers that do not need torch or tf do not
    pay for an import or fail on a missing optional dependency.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except RuntimeError:
            # CUDA backends may refuse strict determinism; warn-only is fine.
            pass
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except AttributeError:
            pass
