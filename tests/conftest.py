import pytest
import numpy as np

from tests.framework.backend import CPUReference, OpenCLBackend


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "opencl: mark test as requiring an OpenCL device"
    )


@pytest.fixture(scope="session")
def reference() -> CPUReference:
    """Shared NumPy reference backend for the entire test session."""
    return CPUReference()


@pytest.fixture(scope="session")
def opencl() -> OpenCLBackend:
    """Shared OpenCL backend. Skips all dependent tests if no device is available."""
    try:
        return OpenCLBackend()
    except Exception as e:
        pytest.skip(f"No OpenCL device available: {e}")
