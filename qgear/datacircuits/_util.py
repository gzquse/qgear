#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def yields_to_pdf(yields, nqubits, normalize=False):
    '''The yields are  expected as a list of dictionaries in the standard
    qiskit format. The number of qubits is the length of each bitstring.
    The normalize flag indicates if sum to 1 or not.'''
    if isinstance(yields, dict):
        yields = [yields]
    ''' Notes for me:  yields is a list containing a dictionary. Len(yields)=1
    It contains 1 dictionary which the # of shots for the qubits - the keys of
    the dictionary are the bit strings or the qubits'''
    pdfs = np.empty([2**nqubits, len(yields)])
    for i, y in enumerate(yields):
        pdfs[:, i] = convert_shots_to_pdf(y, normalize=normalize)
    return pdfs


def con_to_ang(data, nq_addr, nq_data):
    '''
    Changing the pixel values to bit string and collecting and grouping the bit
    values from under the various bit positions in the bit string.  So group
    all the 1st bit postion values from the bit strings generated from the
    pixel values under one column vector. For bit position i in all the pixels,
    we record those under column i in the matrix generated
    '''
    th = np.empty(shape=(2**nq_addr, nq_data))
    # column i contain bit 'i' for bitstrings of all pixel values
    # row j contains the bit string for pixel value for the jth pixel value
    for k in range(nq_data):
        tmps = []
        for i in range(2**nq_addr):
            tmp = get_bit(data[i], k)
            tmps = np.append(tmps, tmp)
        th[:, k] = tmps
    return th


def rescale_angles_to_bit_to_data(angles):
    ''' Converts angles back to bits and then to pixel data values
        Arg: angles
            2-D np array of size n 2^nq_addr by bit depth (2^nq_data)
    '''
    angles = np.where(angles != 0, 1, angles)  # back to bit strings
    angles = angles.astype(int)
    angles = angles.astype(str)
    img = np.empty(shape=angles.shape[0])
    for i in range(angles.shape[0]):
        tmp = angles[i, :]
        b = ""
        for j in range(len(tmp)):
            tp = str(tmp[j])
            b = b + tp
        b = b[::-1]
        img[i] = int(b, 2)
        img = img.astype(int)
    return img


def rescale_bits_to_angle(data):
    ''' Converts bits to angles'''
    ''' data is a numpy array containing 0's and 1's'''
    angles = np.where(data == 1, np.pi/2, data)
    return angles


def next_pow2(x):
    return 1 << (x-1).bit_length()


def circular_bit_shift(b, shift, n):
    '''A circular bit shift with `shift` on the last n bits of b

    Args:
        b: int:
            input integer
        shift: int
            shift parameter in [0, ..., n-1]
        n: int
            number of least bits of b to be shifted
    '''
    rotated = (b >> shift) | (b << n - shift)
    return rotated & (2**n - 1)


def gray_code(b):
    '''Gray code of b.
    Args:
        b: int:
            binary integer
    Returns:
        Gray code of b.
    '''
    return b ^ (b >> 1)


def shifted_gray_code(b, shift, n):
    '''Shifted Gray code of b for use with QCrank'''
    return circular_bit_shift(gray_code(b), shift, n)


def gray_permutation(a):
    '''Permute the vector a from binary to Gray code order with optional shift

    Args:
        a: vectors
            2D Numpy array of size (k, 2**n) -- k can be 1.
    Returns:
        vectors:
            (rowwise) Gray code permutation of vectors a
    '''
    N = a.shape[0]
    b = np.empty(a.shape)
    for i in range(N):
        b[i] = a[gray_code(i)]
    return b


def inv_gray_permutation(a):
    '''Permute the vector a from binary to Gray code order with optional shift

    Args:
        a: vectors
            2D Numpy array of size (k, 2**n) -- k can be 1.
    Returns:
        vectors:
            (rowwise) inverse Gray code permutation of vectors a
    '''
    N = a.shape[0]
    b = np.empty(a.shape)
    for i in range(N):
        b[gray_code(i)] = a[i]
    return b


def shifted_gray_permutation(a, shift):
    '''Shifted Gray code permutation for use with QCrank
    Args:
        a: vector
        shift: integer shift
    '''
    N = a.shape[0]
    n = int(np.log2(N))
    b = np.empty(a.shape)
    for i in range(N):
        b[i] = a[shifted_gray_code(i, shift, n)]
    return b


def shifted_inv_gray_permutation(a, shift):
    '''Shifted inverse Gray code permutation for use with QCrank.
    # TODO vectorize (3Dimnsions instances x data x addr) should be looped
    over the data dimension to change shift
    Args:
        a: vector
        shift: integer shift
    '''
    N = a.shape[0]
    n = int(np.log2(N))
    b = np.zeros(N)
    for i in range(N):
        b[shifted_gray_code(i, shift, n)] = a[i]
    return b


def sfwht(a):
    '''Scaled Fast Walsh-Hadamard transform of input vectors a.

    Args:
        a: vectors
            2D Numpy array of size (k, 2**n) -- k can be 1.
    Returns:
        vectors:
            Scaled Walsh-Hadamard transform of vectors a.
    '''
    N = a.shape[0]
    b = np.copy(a)
    n = int(np.log2(N))
    for h in range(n):
        for i in range(0, N, 2**(h+1)):
            for j in range(i, i+2**h):
                x = np.copy(b[j])  # need to copy explicitly
                y = b[j + 2**h]
                b[j] = (x + y) / 2.
                b[j + 2**h] = (x - y) / 2.
    return b


def isfwht(a):
    '''Inverse scaled Fast Walsh-Hadamard transform of input vectors a.

    Args:
        a: vectors
            2D Numpy array of size (k, 2**n) -- k can be 1.
    Returns:
        vectors:
            inverse scaled Walsh-Hadamard transform of vectors a.
    '''
    N = a.shape[0]
    b = np.copy(a)
    n = int(np.log2(N))
    for h in range(n):
        for i in range(0, N, 2**(h+1)):
            for j in range(i, i+2**h):
                x = np.copy(b[j])  # need to copy explicitly
                y = b[j + 2**h]
                b[j] = x + y
                b[j + 2**h] = x - y
    return b


def compute_control(i, n, shift=0):
    '''Compute the control qubit index based on the index i and size n. An
    optional shift can be used to vertically shift the CNOT cycle (mod n).'''
    return int(
        (n + shift - 1 -
            np.log2(gray_code((i+1) % 2**n) ^ gray_code(i % 2**n))) % n
        )


def rescale_data_to_angles(data, max_val=256, flatten_and_pad=False):
    '''Takes in the data, flattens it, converts it to angles, and applies the
    permuted FWHT to get the agles that are ready for the FRQI circuit.

    Args:
        data:
            numerical numpy array of data with values in interval
            [0, max_val-1]
        max_val: int (256)
            maximum intensity value (default: 256 for 8bit). The maximum
            intensity is non-inclusive, i.e., intensities in
            [0, ..., max_val - 1] are expected.
        flatten_and_pad: bool
            indicates whether the data is flattened to a 1D array and zero
            padded to the next power of two or not.
    Returns:
        angles:
            the angles corresponding to the data
    '''
    # convert to angles determined by max_val
    pi = np.pi
    sc = pi / max_val
    angles = np.clip(data * sc, 0, pi)
    if flatten_and_pad:
        return np.pad(
            np.ravel(angles), (0, next_pow2(angles.shape[0]) - angles.shape[0])
        )
    return angles

def rescale_angles_to_fdata(angles, max_val=256):
    ''' Converts the angles back to discretized data according to a linear
    relation.
        
    Args:
        angles:
            measured/computed angles
        max_val: float (256)
            maximum intensity value can be any positive real number.
            intensities in [0, max_val) are expected.
    ''' 
    pi = np.pi
    sc = max_val / pi
    return angles * sc
      

def convert_shots_to_pdf(counts, normalize=True):
    '''Converts the counts as returned by Qiskit to a probability density
    function.

    Args:
        counts: dict
            Dictionary returned by Qiskit containing the counts in the run.
        normalize:
            normalize PDF to 1 or not
    '''
    nqubits = len(list(counts.keys())[0])
    nshots = sum(counts.values())
    N = 2**nqubits
    pdf = np.zeros(N)
    for i in range(N):
        key = format(i, 'b')
        key = '0' * (nqubits - len(key)) + key
        try:
            pdf[i] = counts[key]
        except KeyError:
            pdf[i] = 0
    return pdf / nshots if normalize else pdf


# bit manipulations for efficient permutations
def get_bit(value, bit):
    return value >> bit & 1


def clear_bit(value, bit):
    return value & ~(1 << bit)


def set_bit(value, bit):
    return value | (1 << bit)


def get_bits(value, bits):
    '''Gets the values of the bits at position `bits` in `value`.

        !!! note
        The bits are assumed to be sorted in ascending order.
    '''
    res = 0
    c = 0
    for b in bits:
        if get_bit(value, b):
            res = set_bit(res, c)
        c += 1
    return res


def cnot_permutation(dist, control, target):
    '''Classically applies a CNOT gate to a probability density function or
    state vector.

    Args:
        dist:
            probability density function, distribution ( yields / shots ), or
            statevector on n qubits
        control: int
            control qubit (< n)
        target: int
            target qubit ( < n and different from control)
    '''
    N = dist.shape[0]
    n = int(np.log2(N))
    result = np.copy(dist)
    for i in range(N):
        if get_bit(i, n-control-1):
            j = clear_bit(i, n-target-1) if get_bit(i, n-target-1) else \
                set_bit(i, n-target-1)
            if i < j:
                result[i] = dist[j]
                result[j] = dist[i]
    return result


def marginal_distribution(dist, trace_out):
    '''Computes a marginal distribution  where a subsystem is traced-out and
    the remaining distribution is returned. Supports vectorized operations.

    Args:
        dist:
            probability density function, distribution (yields / shots) on n
            qubits -- s.
        trace_out:
            array of qubits in [0, ..., n-1] to trace out from the distribution
    Returns:
        marginal distribution on the remaining qubits.
    '''
    N = dist.shape[0]
    n = int(np.log2(N))
    m = len(trace_out)
    k = n - m
    dist_traced = np.zeros((2**k, *dist.shape[1:]))
    trace_out = [(n - 1 - t) % n for t in trace_out]
    trace_bits = set(trace_out)
    all_bits = set([i for i in range(n)])
    keep_bits = sorted(list(all_bits - trace_bits))

    for i in range(N):
        k = get_bits(i, keep_bits)
        dist_traced[k] += dist[i]
    return dist_traced