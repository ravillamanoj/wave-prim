"""Property-based correctness tests for the Reduce primitive."""

import numpy as np
import pytest

from tests.framework import assert_allclose
from tests.framework.runner import PrimitiveTestBase, TEST_SIZES


class TestReduceProperties(PrimitiveTestBase):
    """Algebraic properties verified against the CPU reference."""

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_sum_matches_numpy(self, reference, size):
        data = self.random_float32(size)
        assert reference.reduce(data, "sum") == pytest.approx(float(np.sum(data)), rel=1e-4)

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_min_matches_numpy(self, reference, size):
        data = self.random_float32(size)
        assert reference.reduce(data, "min") == pytest.approx(float(np.min(data)), rel=1e-4)

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_max_matches_numpy(self, reference, size):
        data = self.random_float32(size)
        assert reference.reduce(data, "max") == pytest.approx(float(np.max(data)), rel=1e-4)

    def test_single_element_sum(self, reference):
        data = np.array([42.0], dtype=np.float32)
        assert reference.reduce(data, "sum") == pytest.approx(42.0)

    def test_single_element_min(self, reference):
        data = np.array([42.0], dtype=np.float32)
        assert reference.reduce(data, "min") == pytest.approx(42.0)

    def test_all_zeros(self, reference):
        data = self.zeros(256)
        assert reference.reduce(data, "sum") == pytest.approx(0.0)

    def test_all_same_value(self, reference):
        data = np.full(128, 3.0, dtype=np.float32)
        assert reference.reduce(data, "sum") == pytest.approx(128 * 3.0, rel=1e-4)

    def test_decomposability(self, reference):
        # reduce(A || B) == reduce(A) op reduce(B)
        a = self.random_float32(64, seed=1)
        b = self.random_float32(64, seed=2)
        combined = np.concatenate([a, b])
        assert reference.reduce(combined, "sum") == pytest.approx(
            reference.reduce(a, "sum") + reference.reduce(b, "sum"), rel=1e-4
        )

    def test_min_is_le_all_elements(self, reference):
        data = self.random_float32(256)
        result = reference.reduce(data, "min")
        assert np.all(result <= data)

    def test_max_is_ge_all_elements(self, reference):
        data = self.random_float32(256)
        result = reference.reduce(data, "max")
        assert np.all(result >= data)

    def test_negative_values(self, reference):
        data = np.array([-5.0, -3.0, -1.0, -4.0], dtype=np.float32)
        assert reference.reduce(data, "min") == pytest.approx(-5.0)
        assert reference.reduce(data, "max") == pytest.approx(-1.0)
        assert reference.reduce(data, "sum") == pytest.approx(-13.0, rel=1e-4)


class TestReduceOpenCL(PrimitiveTestBase):
    """OpenCL kernel output matches CPU reference across all sizes and operations."""

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_sum(self, reference, opencl, size):
        data = self.random_float32(size)
        assert opencl.reduce(data, "sum") == pytest.approx(
            reference.reduce(data, "sum"), rel=1e-3
        )

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_min(self, reference, opencl, size):
        data = self.random_float32(size)
        assert opencl.reduce(data, "min") == pytest.approx(
            reference.reduce(data, "min"), rel=1e-4
        )

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_max(self, reference, opencl, size):
        data = self.random_float32(size)
        assert opencl.reduce(data, "max") == pytest.approx(
            reference.reduce(data, "max"), rel=1e-4
        )

    def test_sum_single_element(self, reference, opencl):
        data = np.array([7.5], dtype=np.float32)
        assert opencl.reduce(data, "sum") == pytest.approx(
            reference.reduce(data, "sum"), rel=1e-5
        )

    def test_sum_all_zeros(self, opencl):
        data = self.zeros(512)
        assert opencl.reduce(data, "sum") == pytest.approx(0.0, abs=1e-6)

    def test_sum_large_array(self, reference, opencl):
        # Exercises multiple workgroups
        data = self.random_float32(4096, seed=99)
        assert opencl.reduce(data, "sum") == pytest.approx(
            reference.reduce(data, "sum"), rel=1e-3
        )

    def test_min_negative_values(self, reference, opencl):
        data = np.array([-10.0, -2.0, -7.0, -1.0], dtype=np.float32)
        assert opencl.reduce(data, "min") == pytest.approx(
            reference.reduce(data, "min"), rel=1e-5
        )

    def test_max_negative_values(self, reference, opencl):
        data = np.array([-10.0, -2.0, -7.0, -1.0], dtype=np.float32)
        assert opencl.reduce(data, "max") == pytest.approx(
            reference.reduce(data, "max"), rel=1e-5
        )

    def test_decomposability(self, reference, opencl):
        a = self.random_float32(128, seed=10)
        b = self.random_float32(128, seed=20)
        combined = np.concatenate([a, b])
        assert opencl.reduce(combined, "sum") == pytest.approx(
            opencl.reduce(a, "sum") + opencl.reduce(b, "sum"), rel=1e-3
        )
