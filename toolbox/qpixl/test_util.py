#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._util import (
    next_pow2,
    circular_bit_shift,
    gray_code,
    shifted_gray_code,
    gray_permutation,
    inv_gray_permutation,
    shifted_gray_permutation,
    shifted_inv_gray_permutation,
    sfwht,
    isfwht,
    compute_control,
    rescale_data_to_angles,
    rescale_angles_to_data,
    convert_shots_to_pdf,
    cnot_permutation,
    marginal_distribution
)
import numpy as np


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(4) == 4
    assert next_pow2(5) == 8
    assert next_pow2(6) == 8
    assert next_pow2(7) == 8
    assert next_pow2(8) == 8
    assert next_pow2(255) == 256
    assert next_pow2(257) == 512


def test_circular_bit_shift():
    # shift 0
    assert circular_bit_shift(0, 0, 3) == 0
    assert circular_bit_shift(1, 0, 3) == 1
    assert circular_bit_shift(2, 0, 3) == 2
    assert circular_bit_shift(3, 0, 3) == 3
    assert circular_bit_shift(4, 0, 3) == 4
    assert circular_bit_shift(5, 0, 3) == 5
    assert circular_bit_shift(6, 0, 3) == 6
    assert circular_bit_shift(7, 0, 3) == 7
    # shift 1
    assert circular_bit_shift(0, 1, 3) == 0
    assert circular_bit_shift(1, 1, 3) == 4
    assert circular_bit_shift(2, 1, 3) == 1
    assert circular_bit_shift(3, 1, 3) == 5
    assert circular_bit_shift(4, 1, 3) == 2
    assert circular_bit_shift(5, 1, 3) == 6
    assert circular_bit_shift(6, 1, 3) == 3
    assert circular_bit_shift(7, 1, 3) == 7
    # shift 2
    assert circular_bit_shift(0, 2, 3) == 0
    assert circular_bit_shift(1, 2, 3) == 2
    assert circular_bit_shift(2, 2, 3) == 4
    assert circular_bit_shift(3, 2, 3) == 6
    assert circular_bit_shift(4, 2, 3) == 1
    assert circular_bit_shift(5, 2, 3) == 3
    assert circular_bit_shift(6, 2, 3) == 5
    assert circular_bit_shift(7, 2, 3) == 7


def test_gray_code():
    assert gray_code(0) == 0
    assert gray_code(1) == 1
    assert gray_code(2) == 3
    assert gray_code(3) == 2
    assert gray_code(4) == 6
    assert gray_code(5) == 7
    assert gray_code(6) == 5
    assert gray_code(7) == 4
    assert gray_code(255) == 128
    assert gray_code(511) == 256


def test_shifted_gray_code():
    # shift 0
    assert shifted_gray_code(0, 0, 3) == 0
    assert shifted_gray_code(1, 0, 3) == 1
    assert shifted_gray_code(2, 0, 3) == 3
    assert shifted_gray_code(3, 0, 3) == 2
    assert shifted_gray_code(4, 0, 3) == 6
    assert shifted_gray_code(5, 0, 3) == 7
    assert shifted_gray_code(6, 0, 3) == 5
    assert shifted_gray_code(7, 0, 3) == 4
    # shift 1
    assert shifted_gray_code(0, 1, 3) == 0
    assert shifted_gray_code(1, 1, 3) == 4
    assert shifted_gray_code(2, 1, 3) == 5
    assert shifted_gray_code(3, 1, 3) == 1
    assert shifted_gray_code(4, 1, 3) == 3
    assert shifted_gray_code(5, 1, 3) == 7
    assert shifted_gray_code(6, 1, 3) == 6
    assert shifted_gray_code(7, 1, 3) == 2
    # shift 2
    assert shifted_gray_code(0, 2, 3) == 0
    assert shifted_gray_code(1, 2, 3) == 2
    assert shifted_gray_code(2, 2, 3) == 6
    assert shifted_gray_code(3, 2, 3) == 4
    assert shifted_gray_code(4, 2, 3) == 5
    assert shifted_gray_code(5, 2, 3) == 7
    assert shifted_gray_code(6, 2, 3) == 3
    assert shifted_gray_code(7, 2, 3) == 1


def test_gray_permutation():
    # single vector
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    b = np.ravel(gray_permutation(a))
    for i, bi in enumerate(b):
        assert bi == a[gray_code(i)]
    c = np.ravel(inv_gray_permutation(b))
    np.testing.assert_array_equal(a, c)
    # two vectors
    a = np.stack(
        [np.array([0, 1, 2, 3, 4, 5, 6, 7]),
         np.array([8, 9, 10, 11, 12, 13, 14, 15])], axis=1
    )
    b = gray_permutation(a)
    for i in range(a.shape[0]):
        np.testing.assert_array_equal(b[i, ...], a[gray_code(i), ...])
    c = inv_gray_permutation(b)
    for i in range(a.shape[0]):
        np.testing.assert_array_equal(a[i, :], c[i, :])


def test_shifted_gray_permutation():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # shift 0
    r = np.array([0, 1, 3, 2, 6, 7, 5, 4])
    b = shifted_gray_permutation(a, 0)
    np.testing.assert_array_equal(b, r)
    c = shifted_inv_gray_permutation(b, 0)
    np.testing.assert_array_equal(a, c)
    # shift 1
    r = np.array([0, 4, 5, 1, 3, 7, 6, 2])
    b = shifted_gray_permutation(a, 1)
    np.testing.assert_array_equal(b, r)
    c = shifted_inv_gray_permutation(b, 1)
    np.testing.assert_array_equal(a, c)
    # shift 2
    r = np.array([0, 2, 6, 4, 5, 7, 3, 1])
    b = shifted_gray_permutation(a, 2)
    np.testing.assert_array_equal(b, r)
    c = shifted_inv_gray_permutation(b, 2)
    np.testing.assert_array_equal(a, c)


def test_sfwht():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float64)
    b = sfwht(a)
    b_ref = np.array([3.5, -0.5, -1, 0, -2, 0, 0, 0], dtype=np.float64)
    print(f'b: {b}')
    print(f'a: {a}')
    np.testing.assert_allclose(b, b_ref, rtol=1e-12)
    a_rec = isfwht(b)
    np.testing.assert_allclose(a_rec, a, rtol=1e-12)


def test_compute_control():
    ctrl_ref = [2, 1, 2, 0, 2, 1, 2, 0]
    shift = [0, 1, 2]
    for i in range(8):
        for s in shift:
            assert compute_control(i, 3, s) == (ctrl_ref[i] + s) % 3


def test_convert_data_angles():
    data = np.array([16., 45., 32., 0., 63., 22., 51., 7.])
    angles = rescale_data_to_angles(data, max_val=64)
    data_rec = rescale_angles_to_data(angles, max_val=64)
    np.testing.assert_allclose(data, data_rec, rtol=1e-12)


def test_convert_shots_to_pdf():
    shots = {'00': 4, '10': 16}
    ref = [4, 0, 16, 0]
    ref_norm = [r / sum(ref) for r in ref]
    pdf = convert_shots_to_pdf(shots, normalize=False)
    np.testing.assert_array_equal(pdf, ref)
    pdf = convert_shots_to_pdf(shots, normalize=True)
    np.testing.assert_allclose(pdf, ref_norm, rtol=1e-12)


def test_cnot_permutation():
    # 2 qubits
    dist = np.array([0, 1, 2, 3])
    dist01 = cnot_permutation(dist, 0, 1)
    np.testing.assert_array_equal(dist01, np.array([0, 1, 3, 2]))
    dist10 = cnot_permutation(dist, 1, 0)
    np.testing.assert_array_equal(dist10, np.array([0, 3, 2, 1]))
    # 3 qubits
    dist = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    dist10 = cnot_permutation(dist, 1, 0)
    np.testing.assert_array_equal(dist10, np.array([0, 1, 6, 7, 4, 5, 2, 3]))
    dist21 = cnot_permutation(dist, 2, 1)
    np.testing.assert_array_equal(dist21, np.array([0, 3, 2, 1, 4, 7, 6, 5]))
    dist02 = cnot_permutation(dist, 0, 2)
    np.testing.assert_array_equal(dist02, np.array([0, 1, 2, 3, 5, 4, 7, 6]))


def test_marginal_distribution():
    # 2 qubits
    dist = np.array([0, 1, 5, 2])
    dist0 = marginal_distribution(dist, [0, ])
    np.testing.assert_array_equal(dist0, np.array([5, 3]))
    dist1 = marginal_distribution(dist, [1, ])
    np.testing.assert_array_equal(dist1, np.array([1, 7]))
    dist01 = marginal_distribution(dist, [0, 1])
    np.testing.assert_array_equal(dist01, np.array([8, ]))
    # 3 qubits
    dist = np.array([1, 4, 5, 2, 2, 6, 3, 1])
    dist0 = marginal_distribution(dist, [0, ])
    ref0 = np.array([3, 10, 8, 3])
    np.testing.assert_array_equal(dist0, ref0)
    dist1 = marginal_distribution(dist, [1, ])
    ref1 = np.array([6, 6, 5, 7])
    np.testing.assert_array_equal(dist1, ref1)
    dist2 = marginal_distribution(dist, [2, ])
    ref2 = np.array([5, 7, 8, 4])
    np.testing.assert_array_equal(dist2, ref2)
    dist01 = marginal_distribution(dist, [0, 1])
    ref01 = np.array([11, 13])
    np.testing.assert_array_equal(dist01, ref01)
    dist02 = marginal_distribution(dist, [0, 2])
    ref02 = np.array([13, 11])
    np.testing.assert_array_equal(dist02, ref02)
    dist12 = marginal_distribution(dist, [1, 2])
    ref12 = np.array([12, 12])
    np.testing.assert_array_equal(dist12, ref12)
    dist012 = dist01 = marginal_distribution(dist, [0, 1, 2])
    ref012 = np.array([24])
    np.testing.assert_array_equal(dist012, ref012)
