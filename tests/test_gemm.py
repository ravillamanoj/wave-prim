"""Property-based correctness tests for the GEMM primitive."""

import numpy as np
import pytest

from tests.framework import assert_allclose
from tests.framework.runner import PrimitiveTestBase

# Tile-aligned, sub-tile, and multi-tile square dimensions
GEMM_SIZES = [4, 8, 16, 32, 64]


class TestGemmProperties(PrimitiveTestBase):
    """Algebraic properties of matrix multiply verified against the CPU reference."""

    @pytest.mark.parametrize("n", GEMM_SIZES)
    def test_identity_right(self, reference, n):
        A = self.random_matrix(n, n)
        I = self.identity_matrix(n)
        assert_allclose(A, reference.gemm(A, I))

    @pytest.mark.parametrize("n", GEMM_SIZES)
    def test_identity_left(self, reference, n):
        A = self.random_matrix(n, n)
        I = self.identity_matrix(n)
        assert_allclose(A, reference.gemm(I, A))

    def test_zero_matrix(self, reference):
        A = self.random_matrix(8, 8)
        Z = np.zeros((8, 8), dtype=np.float32)
        assert_allclose(Z, reference.gemm(A, Z))
        assert_allclose(Z, reference.gemm(Z, A))

    def test_transpose_property(self, reference):
        A = self.random_matrix(5, 7, seed=1)
        B = self.random_matrix(7, 3, seed=2)
        AB = reference.gemm(A, B)
        BtAt = reference.gemm(B.T.copy(), A.T.copy())
        assert_allclose(AB.T, BtAt)

    def test_output_shape(self, reference):
        A = self.random_matrix(3, 5)
        B = self.random_matrix(5, 2)
        assert reference.gemm(A, B).shape == (3, 2)

    def test_associativity(self, reference):
        A = self.random_matrix(4, 4, seed=1)
        B = self.random_matrix(4, 4, seed=2)
        C = self.random_matrix(4, 4, seed=3)
        assert_allclose(
            reference.gemm(reference.gemm(A, B), C),
            reference.gemm(A, reference.gemm(B, C)),
            rtol=1e-3,
        )


class TestGemmOpenCL(PrimitiveTestBase):
    """OpenCL tiled GEMM output matches CPU reference."""

    @pytest.mark.parametrize("n", GEMM_SIZES)
    def test_square_matches_reference(self, reference, opencl, n):
        A = self.random_matrix(n, n, seed=10)
        B = self.random_matrix(n, n, seed=20)
        assert_allclose(reference.gemm(A, B), opencl.gemm(A, B), rtol=1e-3)

    @pytest.mark.parametrize("n", GEMM_SIZES)
    def test_output_shape(self, opencl, n):
        A = self.random_matrix(n, n)
        B = self.random_matrix(n, n)
        assert opencl.gemm(A, B).shape == (n, n)

    def test_identity_right(self, reference, opencl):
        A = self.random_matrix(32, 32)
        I = self.identity_matrix(32)
        assert_allclose(A, opencl.gemm(A, I), rtol=1e-4)

    def test_identity_left(self, reference, opencl):
        A = self.random_matrix(32, 32)
        I = self.identity_matrix(32)
        assert_allclose(A, opencl.gemm(I, A), rtol=1e-4)

    def test_zero_matrix(self, opencl):
        A = self.random_matrix(16, 16)
        Z = np.zeros((16, 16), dtype=np.float32)
        result = opencl.gemm(A, Z)
        assert_allclose(Z, result, rtol=1e-6, atol=1e-6)

    def test_non_square(self, reference, opencl):
        # 3x7 times 7x5 = 3x5; all dims are sub-tile
        A = self.random_matrix(3, 7, seed=1)
        B = self.random_matrix(7, 5, seed=2)
        assert opencl.gemm(A, B).shape == (3, 5)
        assert_allclose(reference.gemm(A, B), opencl.gemm(A, B), rtol=1e-3)

    def test_non_tile_multiple(self, reference, opencl):
        # 17x13 times 13x19: none of the dims are multiples of TILE_SIZE=16
        A = self.random_matrix(17, 13, seed=3)
        B = self.random_matrix(13, 19, seed=4)
        assert opencl.gemm(A, B).shape == (17, 19)
        assert_allclose(reference.gemm(A, B), opencl.gemm(A, B), rtol=1e-3)

    def test_tall_skinny(self, reference, opencl):
        A = self.random_matrix(64, 4, seed=5)
        B = self.random_matrix(4, 64, seed=6)
        assert_allclose(reference.gemm(A, B), opencl.gemm(A, B), rtol=1e-3)

    def test_large(self, reference, opencl):
        A = self.random_matrix(128, 128, seed=7)
        B = self.random_matrix(128, 128, seed=8)
        assert_allclose(reference.gemm(A, B), opencl.gemm(A, B), rtol=1e-3)

    def test_transpose_property(self, reference, opencl):
        A = self.random_matrix(8, 12, seed=9)
        B = self.random_matrix(12, 6, seed=10)
        AB_gpu = opencl.gemm(A, B)
        BtAt_ref = reference.gemm(B.T.copy(), A.T.copy())
        assert_allclose(AB_gpu.T, BtAt_ref, rtol=1e-3)
