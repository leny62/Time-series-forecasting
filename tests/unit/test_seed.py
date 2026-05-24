import os
import random

import numpy as np

from mtraffic.utils import seed


def test_set_all_is_idempotent_for_python_random() -> None:
    seed.set_all(20251201)
    a = [random.random() for _ in range(5)]
    seed.set_all(20251201)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_all_seeds_numpy() -> None:
    seed.set_all(20251201)
    a = np.random.rand(5)
    seed.set_all(20251201)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_pythonhashseed_set() -> None:
    seed.set_all(20251201)
    assert os.environ["PYTHONHASHSEED"] == "20251201"
