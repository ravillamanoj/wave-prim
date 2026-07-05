import numpy as np


# Default tolerances for float32 OpenCL vs CPU comparison.
# Parallel reductions accumulate rounding differently from sequential CPU ops.
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-6


def assert_allclose(
    expected: np.ndarray,
    actual: np.ndarray,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    label: str = "",
) -> None:
    """Assert two arrays are numerically close within relative and absolute tolerance."""
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"Numerical mismatch{f' in {label}' if label else ''}"
    )


def assert_exact(
    expected: np.ndarray,
    actual: np.ndarray,
    label: str = "",
) -> None:
    """Assert two arrays are exactly equal — used for integer operations."""
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=f"Exact match failed{f' in {label}' if label else ''}"
    )


def assert_sorted(arr: np.ndarray) -> None:
    """Assert array elements are in non-decreasing order."""
    assert np.all(arr[:-1] <= arr[1:]), (
        f"Array is not sorted. First violation at index "
        f"{np.argmax(arr[:-1] > arr[1:])}: {arr}"
    )


def assert_permutation(original: np.ndarray, result: np.ndarray) -> None:
    """Assert result contains exactly the same elements as original (no drops, no additions)."""
    assert np.array_equal(np.sort(original.ravel()), np.sort(result.ravel())), (
        "Result is not a permutation of the input — elements were added or dropped"
    )
