"""Property-based correctness tests for the Scan primitive."""

import numpy as np
import pytest

from tests.framework import assert_allclose, assert_exact
from tests.framework.runner import PrimitiveTestBase, TEST_SIZES


class TestScanProperties(PrimitiveTestBase):
    """Mathematical properties of prefix scan verified against the CPU reference."""

    def test_exclusive_first_element_is_zero(self, reference):
        data = self.random_float32(64)
        result = reference.scan(data, inclusive=False)
        assert result[0] == pytest.approx(0.0)

    def test_inclusive_last_equals_total_sum(self, reference):
        data = self.random_float32(256)
        result = reference.scan(data, inclusive=True)
        assert result[-1] == pytest.approx(float(np.sum(data)), rel=1e-4)

    def test_inclusive_minus_input_equals_exclusive(self, reference):
        data = self.random_float32(128)
        inclusive = reference.scan(data, inclusive=True)
        exclusive = reference.scan(data, inclusive=False)
        assert_allclose(inclusive - data, exclusive)

    def test_zeros_inclusive(self, reference):
        data = self.zeros(64)
        assert_exact(self.zeros(64), reference.scan(data, inclusive=True))

    def test_zeros_exclusive(self, reference):
        data = self.zeros(64)
        assert_exact(self.zeros(64), reference.scan(data, inclusive=False))

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_inclusive_matches_numpy(self, reference, size):
        data = self.random_float32(size)
        # Use float64 cumsum to match reference precision
        expected = np.cumsum(data.astype(np.float64)).astype(np.float32)
        assert_allclose(expected, reference.scan(data, inclusive=True))

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_exclusive_matches_numpy(self, reference, size):
        data = self.random_float32(size)
        expected = np.concatenate([[0.0], np.cumsum(data.astype(np.float64))[:-1]]).astype(np.float32)
        assert_allclose(expected, reference.scan(data, inclusive=False))


class TestScanOpenCL(PrimitiveTestBase):
    """OpenCL scan kernel matches CPU reference across all sizes."""

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_exclusive(self, reference, opencl, size):
        data = self.random_float32(size)
        # Float32 scan accumulates rounding error O(n*eps); use rtol=1e-3 for large n
        rtol = 1e-3 if size >= 1024 else 1e-4
        assert_allclose(
            reference.scan(data, inclusive=False),
            opencl.scan(data, inclusive=False),
            rtol=rtol,
        )

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_inclusive(self, reference, opencl, size):
        data = self.random_float32(size)
        rtol = 1e-3 if size >= 1024 else 1e-4
        assert_allclose(
            reference.scan(data, inclusive=True),
            opencl.scan(data, inclusive=True),
            rtol=rtol,
        )

    def test_exclusive_first_element_zero(self, opencl):
        data = self.random_float32(64)
        result = opencl.scan(data, inclusive=False)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_inclusive_last_equals_sum(self, reference, opencl):
        data = self.random_float32(512)
        result = opencl.scan(data, inclusive=True)
        assert result[-1] == pytest.approx(float(np.sum(data)), rel=1e-3)

    def test_zeros(self, opencl):
        data = self.zeros(256)
        assert_exact(self.zeros(256), opencl.scan(data, inclusive=False))

    def test_multi_block_exclusive(self, reference, opencl):
        # 1200 elements > 512 per block — exercises offset correction across 3 blocks
        data = self.random_float32(1200, seed=7)
        assert_allclose(
            reference.scan(data, inclusive=False),
            opencl.scan(data, inclusive=False),
            rtol=1e-3,
        )

    def test_multi_block_inclusive(self, reference, opencl):
        data = self.random_float32(2048, seed=8)
        assert_allclose(
            reference.scan(data, inclusive=True),
            opencl.scan(data, inclusive=True),
            rtol=1e-3,
        )

    def test_single_element_exclusive(self, reference, opencl):
        data = np.array([3.14], dtype=np.float32)
        assert_allclose(
            reference.scan(data, inclusive=False),
            opencl.scan(data, inclusive=False),
        )

    def test_single_element_inclusive(self, reference, opencl):
        data = np.array([3.14], dtype=np.float32)
        assert_allclose(
            reference.scan(data, inclusive=True),
            opencl.scan(data, inclusive=True),
        )
