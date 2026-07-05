import numpy as np
import pytest

from .backend import CPUReference, OpenCLBackend


# Standard sizes exercised by every primitive test suite.
# Covers edge cases (1, 2), small (16, 64), medium (256), and larger (1024) arrays.
TEST_SIZES = [1, 2, 16, 64, 256, 1024]


@pytest.fixture(scope="session")
def reference() -> CPUReference:
    """Shared NumPy reference backend for the entire test session."""
    return CPUReference()


@pytest.fixture(scope="session")
def opencl() -> OpenCLBackend:
    """Shared OpenCL backend for the entire test session.
    Skips all dependent tests if no OpenCL device is available.
    """
    try:
        return OpenCLBackend()
    except Exception as e:
        pytest.skip(f"No OpenCL device available: {e}")


class PrimitiveTestBase:
    """Base class for primitive correctness test suites.

    Provides data generators and standard test sizes. Subclasses
    write test methods using the `reference` and `opencl` fixtures
    from this module.
    """

    @staticmethod
    def random_float32(size: int, low: float = -100.0, high: float = 100.0, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, size).astype(np.float32)

    @staticmethod
    def random_int32(size: int, low: int = -1000, high: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(low, high, size, dtype=np.int32)

    @staticmethod
    def random_matrix(rows: int, cols: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, (rows, cols)).astype(np.float32)

    @staticmethod
    def zeros(size: int) -> np.ndarray:
        return np.zeros(size, dtype=np.float32)

    @staticmethod
    def ones(size: int) -> np.ndarray:
        return np.ones(size, dtype=np.float32)

    @staticmethod
    def identity_matrix(n: int) -> np.ndarray:
        return np.eye(n, dtype=np.float32)
