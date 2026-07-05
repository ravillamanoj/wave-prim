"""Property-based correctness tests for the Sort primitive."""

import numpy as np
import pytest

from tests.framework import assert_allclose, assert_exact, assert_sorted, assert_permutation
from tests.framework.runner import PrimitiveTestBase, TEST_SIZES


class TestSortProperties(PrimitiveTestBase):
    """Mathematical properties of sort verified against the CPU reference."""

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_output_is_sorted(self, reference, size):
        data = self.random_float32(size)
        assert_sorted(reference.sort(data))

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_output_is_permutation(self, reference, size):
        data = self.random_float32(size)
        assert_permutation(data, reference.sort(data))

    def test_already_sorted(self, reference):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert_exact(data, reference.sort(data))

    def test_reverse_sorted(self, reference):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        result = reference.sort(data)
        assert_sorted(result)
        assert_permutation(data, result)

    def test_all_equal(self, reference):
        data = np.full(64, 7.0, dtype=np.float32)
        result = reference.sort(data)
        assert_sorted(result)
        assert_exact(data, result)

    def test_negative_values(self, reference):
        data = np.array([-3.0, -1.0, -4.0, -1.0, -5.0], dtype=np.float32)
        result = reference.sort(data)
        assert_sorted(result)
        assert result[0] == pytest.approx(-5.0)

    def test_idempotent(self, reference):
        data = self.random_float32(128)
        once = reference.sort(data)
        twice = reference.sort(once)
        assert_exact(once, twice)

    def test_single_element(self, reference):
        data = np.array([42.0], dtype=np.float32)
        result = reference.sort(data)
        assert_exact(data, result)


class TestSortOpenCL(PrimitiveTestBase):
    """OpenCL bitonic sort output matches CPU reference across all sizes."""

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_output_is_sorted(self, opencl, size):
        data = self.random_float32(size)
        assert_sorted(opencl.sort(data))

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_output_is_permutation(self, opencl, size):
        data = self.random_float32(size)
        assert_permutation(data, opencl.sort(data))

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_matches_reference(self, reference, opencl, size):
        data = self.random_float32(size)
        assert_exact(reference.sort(data), opencl.sort(data))

    def test_already_sorted(self, opencl):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        assert_exact(data, opencl.sort(data))

    def test_reverse_sorted(self, opencl):
        data = np.arange(16, 0, -1, dtype=np.float32)
        result = opencl.sort(data)
        assert_sorted(result)
        assert_permutation(data, result)

    def test_all_equal(self, opencl):
        data = np.full(64, 3.14, dtype=np.float32)
        result = opencl.sort(data)
        assert_sorted(result)

    def test_negative_values(self, reference, opencl):
        data = np.array([-5.0, 2.0, -3.0, 8.0, 0.0, -1.0], dtype=np.float32)
        assert_exact(reference.sort(data), opencl.sort(data))

    def test_non_power_of_two(self, reference, opencl):
        # 100 is not a power of two; exercises padding logic
        data = self.random_float32(100, seed=42)
        assert_exact(reference.sort(data), opencl.sort(data))

    def test_large_array(self, reference, opencl):
        data = self.random_float32(4096, seed=99)
        assert_sorted(opencl.sort(data))
        assert_permutation(data, opencl.sort(data))

    def test_single_element(self, reference, opencl):
        data = np.array([7.5], dtype=np.float32)
        assert_exact(reference.sort(data), opencl.sort(data))
