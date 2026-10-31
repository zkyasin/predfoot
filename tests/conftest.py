import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator: every statistical test is reproducible."""
    return np.random.default_rng(20261031)
