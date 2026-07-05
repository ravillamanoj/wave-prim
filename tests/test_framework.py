"""Tests for the testing framework itself — verifies CPUReference and tolerance utilities."""

import numpy as np
import pytest

from tests.framework import (
    CPUReference,
    OpenCLBackend,
    assert_allclose,
    assert_exact,
    assert_sorted,
    assert_permutation,
)
from tests.framework.runner import PrimitiveTestBase, TEST_SIZES


ref = CPUReference()


# ---------------------------------------------------------------------------
# CPUReference — reduce
# ---------------------------------------------------------------------------

class TestCPUReduce:

    def test_sum(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert ref.reduce(data, "sum") == pytest.approx(10.0)

    def test_min(self):
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        assert ref.reduce(data, "min") == pytest.approx(1.0)

    def test_max(self):
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        assert ref.reduce(data, "max") == pytest.approx(5.0)

    def test_single_element(self):
        data = np.array([7.0], dtype=np.float32)
        assert ref.reduce(data, "sum") == pytest.approx(7.0)

    def test_invalid_op(self):
        with pytest.raises(ValueError):
            ref.reduce(np.ones(4, dtype=np.float32), "product")

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_sum_matches_numpy(self, size):
        data = PrimitiveTestBase.random_float32(size)
        assert ref.reduce(data, "sum") == pytest.approx(float(np.sum(data)), rel=1e-5)


# ---------------------------------------------------------------------------
# CPUReference — scan
# ---------------------------------------------------------------------------

class TestCPUScan:

    def test_inclusive(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.array([1.0, 3.0, 6.0, 10.0], dtype=np.float32)
        assert_allclose(expected, ref.scan(data, inclusive=True))

    def test_exclusive(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float32)
        assert_allclose(expected, ref.scan(data, inclusive=False))

    def test_single_element_inclusive(self):
        data = np.array([5.0], dtype=np.float32)
        assert_allclose(np.array([5.0], dtype=np.float32), ref.scan(data))

    def test_last_element_equals_total_sum(self):
        data = PrimitiveTestBase.random_float32(256)
        result = ref.scan(data, inclusive=True)
        assert result[-1] == pytest.approx(float(np.sum(data)), rel=1e-4)

    def test_zeros(self):
        data = PrimitiveTestBase.zeros(64)
        result = ref.scan(data)
        assert_exact(data, result)


# ---------------------------------------------------------------------------
# CPUReference — sort
# ---------------------------------------------------------------------------

class TestCPUSort:

    def test_basic(self):
        data = np.array([4.0, 2.0, 7.0, 1.0, 3.0], dtype=np.float32)
        result = ref.sort(data)
        assert_sorted(result)
        assert_permutation(data, result)

    def test_already_sorted(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = ref.sort(data)
        assert_exact(data, result)

    def test_reverse_sorted(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        result = ref.sort(data)
        assert_sorted(result)

    def test_idempotent(self):
        data = PrimitiveTestBase.random_float32(128)
        once = ref.sort(data)
        twice = ref.sort(once)
        assert_exact(once, twice)

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_permutation_property(self, size):
        data = PrimitiveTestBase.random_float32(size)
        result = ref.sort(data)
        assert_permutation(data, result)
        assert_sorted(result)


# ---------------------------------------------------------------------------
# CPUReference — GEMM
# ---------------------------------------------------------------------------

class TestCPUGemm:

    def test_identity(self):
        A = PrimitiveTestBase.random_matrix(4, 4)
        I = PrimitiveTestBase.identity_matrix(4)
        assert_allclose(A, ref.gemm(A, I))
        assert_allclose(A, ref.gemm(I, A))

    def test_zero_matrix(self):
        A = PrimitiveTestBase.random_matrix(4, 4)
        Z = np.zeros((4, 4), dtype=np.float32)
        result = ref.gemm(A, Z)
        assert_allclose(Z, result)

    def test_transpose_property(self):
        A = PrimitiveTestBase.random_matrix(3, 4)
        B = PrimitiveTestBase.random_matrix(4, 3)
        AB = ref.gemm(A, B)
        BtAt = ref.gemm(B.T, A.T)
        assert_allclose(AB.T, BtAt)

    def test_non_square(self):
        A = PrimitiveTestBase.random_matrix(3, 5)
        B = PrimitiveTestBase.random_matrix(5, 2)
        result = ref.gemm(A, B)
        assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# Tolerance utilities
# ---------------------------------------------------------------------------

class TestToleranceUtilities:

    def test_assert_allclose_passes(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = a + 1e-7
        assert_allclose(a, b)

    def test_assert_allclose_fails_on_large_diff(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = a + 1.0
        with pytest.raises(AssertionError):
            assert_allclose(a, b)

    def test_assert_sorted_passes(self):
        assert_sorted(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def test_assert_sorted_fails(self):
        with pytest.raises(AssertionError):
            assert_sorted(np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32))

    def test_assert_permutation_passes(self):
        original = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        result   = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert_permutation(original, result)

    def test_assert_permutation_fails_on_drop(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result   = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        with pytest.raises(AssertionError):
            assert_permutation(original, result)


# ---------------------------------------------------------------------------
# OpenCLBackend — instantiation only (kernels come in later phases)
# ---------------------------------------------------------------------------

class TestOpenCLBackend:

    def test_backend_instantiates(self, opencl):
        assert opencl is not None

    def test_reduce_executes(self, opencl):
        result = opencl.reduce(np.ones(4, dtype=np.float32), "sum")
        assert result == pytest.approx(4.0)

    def test_scan_executes(self, opencl):
        result = opencl.scan(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), inclusive=False)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_sort_executes(self, opencl):
        result = opencl.sort(np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32))
        assert result[0] == pytest.approx(1.0)

    def test_gemm_executes(self, opencl):
        I = np.eye(4, dtype=np.float32)
        result = opencl.gemm(I, I)
        assert result.shape == (4, 4)
