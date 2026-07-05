import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "opencl: mark test as requiring an OpenCL device"
    )
