#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from ._util import (
    get_bit,
    yields_to_pdf,
    get_bits,
    compute_control,
    shifted_gray_permutation,
    sfwht,
    marginal_distribution,
    con_to_ang,
    rescale_angles_to_bit_to_data,
    rescale_bits_to_angle,
    cnot_permutation
)


class _DecoderNEQR(ABC):
    '''Abstract base class for NEQR data decoders.'''
    def __init__(self, nq_addr, nq_data) -> None:
        self.nq_addr = nq_addr
        self.nq_data = nq_data

    def yields_to_data(self, yields, is_numpy=False):
        if is_numpy:
            assert yields.ndim == 2  # expected shape:  [bitstrings,images]
            pdfs = yields
        else:  # yields is a Qiskit dictionary
            pdfs = yields_to_pdf(yields, self.nq_addr + self.nq_data)
        return pdfs


class NoiseFreeDecoderNEQR(_DecoderNEQR):
    '''Assumes perfect output -- as soon as a bit string is set, it
    is never overwritten.
    '''
    def yields_to_data(self, yields, is_numpy=False):
        pdfs = super().yields_to_data(yields, is_numpy)
        k = pdfs.shape[1]
        data = np.zeros([2**self.nq_addr, k])
        for kk in range(k):
            for i in range(pdfs.shape[0]):
                if pdfs[i, kk] > 0:
                    loc = get_bits(i, list(range(0, self.nq_addr)))
                    if data[loc, kk] == 0:
                        data[loc, kk] = get_bits(
                            i,
                            list(
                                range(
                                    self.nq_addr,
                                    self.nq_data + self.nq_addr
                                )
                            )
                        )
        return data


class NEQR_MCX:
    '''An object to handle circuits for NEQR encodings through multi-controlled
    CX gates.'''
    def __init__(self, nq_addr, nq_data,
                 decoder=NoiseFreeDecoderNEQR,
                 measure: bool = True,
                 statevec: bool = False,
                 reverse_bits: bool = False,
                 barrier: bool = True):
        '''
        Initializes a NEQR circuit with nq_addr qubits and nq_data qubits. The
        data is encoded through multi-controlled CNOT gates.

        Args:
            nq_addr: int
                number of address qubits
            nq_data: int
                number of data qubits
            measure: bool (True)
                adds measurements to all qubits
            statevec: bool (False)
                set to true for state vector simulation
            reverse_bits: bool (False)
                set to true for reversing the order of the bits
            barrier: bool (False)
                add barrier to circuit

        !!! note
            `measure` and `statevec` cannot be simultaneously set to True. If
            both are input as True, only measure will be configured.

        !!! note
            without configuring the output circuits don't have measurements
        '''
        self._nq_addr = nq_addr
        self._nq_data = nq_data
        self.decoder = decoder(nq_addr, nq_data)
        self.measure = measure
        self.barrier = barrier
        if statevec and self.measure:
            raise RuntimeWarning('Ignoring statevec flag over measurement'
                                 'flag')
        else:
            self.statevec = statevec
        self.reverse_bits = reverse_bits

    @property
    def nq_addr(self):
        return self._nq_addr

    @property
    def nq_data(self):
        return self._nq_data

    def generate_from_data(self, data):
        '''Generates the NEQR circuit that encodes the data.

        Args:
            data:
                Integer numerical data with bit-depth nq_data to generate NEQR
                circuits for. The following input options are supported:
                    * numpy array of size (2**nq_addr,)
                    * list of numpy arrays of size (2**nq_addr,)
                    * numpy array of size (2**nq_addr, k)
        '''
        if not isinstance(data, (np.ndarray, list)):
            raise RuntimeError('data should be either numpy array or list of '
                               f'numpy array, got {isinstance(data)}')
        if isinstance(data, list):
            data = np.stack(data, axis=2)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = data[..., np.newaxis]
        if data.shape[0] != 2**self.nq_addr:
            raise RuntimeError(
                f'Input data of incorrect shape {data.shape}, expecting '
                f'({2**self.nq_addr}, ...) '
                '[(2**nq_addr, k)]'
            )
        if np.max(data) >= 2**self.nq_data:
            raise RuntimeError(
                f'Input data not properly normalized, expected data in range '
                f'[0, {2**self.nq_data-1}], found {np.max(data)}.'
            )
        circs = []
        for k in range(data.shape[1]):
            circ = QuantumCircuit(self._nq_addr + self._nq_data)
            # add diffusion
            for i in range(self.nq_addr):
                circ.h(i)
            if self.barrier:
                circ.barrier()
            # add multi-controlled CX gates
            for i, bi in enumerate(data[:, k]):
                # flip zero control bits
                for j in range(self.nq_addr):
                    if not get_bit(i, j):
                        circ.x(j)
                ctrl = list(range(0, self.nq_addr))
                for j in range(self.nq_data):
                    if get_bit(bi, j):
                        circ.mcx(ctrl, self.nq_addr + j)
                # flip zero control bits back
                for j in range(self.nq_addr):
                    if not get_bit(i, j):
                        circ.x(j)
                if self.barrier:
                    circ.barrier()
            if self.reverse_bits:
                circ = circ.reverse_bits()
            if self.measure:
                circ.measure_all()
            if self.statevec:
                circ.save_statevector()
            circs.append(circ)
        return circs


class _DecoderNEQCRANK(ABC):
    '''Abstract base class for NEQR-QCRANK data decoders.'''
    def __init__(self, nq_addr, nq_data, keep_last_cx) -> None:
        self.nq_addr = nq_addr
        self.nq_data = nq_data
        self.keep_last_cx = keep_last_cx

    @abstractmethod
    def angles_from_yields(self, yields):
        pass

    @abstractmethod
    def angles_from_statevec(self, statevec):
        pass

    @staticmethod
    def angles_to_data(angles):
        return rescale_angles_to_bit_to_data(angles)

    def dist_to_marginals(self, dist):
        out = np.empty((2**(self.nq_addr+1), self.nq_data))
        for i in range(self.nq_data):
            t_out = [k + self.nq_addr for k in range(self.nq_data) if k != i]
            md = marginal_distribution(dist, t_out)
            out[:, i] = np.ravel(md)
        return out


class QKAtan2DecoderQCRANK(_DecoderNEQCRANK):
    '''Qiskit compatible atan2 decoder for NEQR-QCRANK.'''
    def angles_from_yields(self, yields, is_numpy=False):
        '''Decodes the angles from the yields of a NEQR-QCRANK experiment.'''
        if is_numpy:
            assert yields.ndim == 2  # expected shape:  [bitstrings,images]
            pdfs = yields
        else:  # yields is a Qiskit dictionary
            pdfs = yields_to_pdf(yields, self.nq_addr + self.nq_data)
        marginal_pdfs = self.dist_to_marginals(pdfs)
        marginal_pdfs = np.sqrt(marginal_pdfs)
        if self.keep_last_cx is False:
            for i in range(self.nq_data):
                marginal_pdfs[:, i] = cnot_permutation(
                    marginal_pdfs[:, i],
                    i % self.nq_addr,
                    self.nq_addr
                )
        return np.arctan2(marginal_pdfs[1::2], marginal_pdfs[::2])

    def angles_from_statevec(self, statevec):
        '''Decodes the angles from the state vector simulation of a QCRANK
        experiment.'''
        if isinstance(statevec, list):
            statevec = np.stack(statevec, axis=1)
        statevec = np.abs(statevec)
        statevec = self.dist_to_marginals(statevec)
        if self.keep_last_cx is False:
            for i in range(self.nq_data):
                statevec[:, i] = cnot_permutation(
                    statevec[:, i],
                    i % self.nq_addr,
                    self.nq_addr
                )
        return np.arctan2(
            statevec[1::2], statevec[::2]
        )


class ParametrizedNEQCRANK:
    '''An object to handle the parametrized NEQR-QCRANK circuits.'''
    def __init__(self, nq_addr, nq_data,
                 decoder: Type[_DecoderNEQCRANK] = QKAtan2DecoderQCRANK,
                 keep_last_cx: bool = True,
                 measure: bool = True,
                 statevec: bool = False,
                 reverse_bits: bool = False):
        '''Initializes a parametrized NEQCRANK circuit with nq_addr qubits and
        nq_data data qubits. Total number of qubits in the circuit is nq_addr +
        nq_data

        Args:
            nq_addr: int
                number of address qubits
            nq_data: int
                number of data qubits
            keep_last_cx: bool (True)
                keep or discard final CX gate
            measure: bool (True)
                adds measurements to all qubits
            statevec: bool (False)
                set to true for state vector simulation
            reverse_bits: bool (False)
                set to true for reversing the order of the bits

        !!! note
            `measure` and `statevec` cannot be simultaneously set to True. If
            both are input as True, only measure will be configured.

        !!! note
            without configuring the output circuits don't have measurements
        '''
        self._nq_addr = nq_addr
        self._nq_data = nq_data
        self.keep_last_cx = keep_last_cx
        self.decoder = decoder(nq_addr, nq_data, keep_last_cx)
        self.measure = measure
        if statevec and self.measure:
            raise RuntimeWarning('Ignoring statevec flag over measurement'
                                 'flag')
        else:
            self.statevec = statevec
        self.reverse_bits = reverse_bits
        # parameter vector
        self._p = [
            ParameterVector(f'p{i}', 2**nq_addr) for i in range(nq_data)
        ]
        # generate circuit
        self.circuit = QuantumCircuit(nq_addr + nq_data)
        # add diffusion
        for i in range(nq_addr):
            self.circuit.h(i)
        self.circuit.barrier()
        # add nested and shifted uniform rotations
        for j in range(2**nq_addr):
            for i in range(nq_data):
                self.circuit.ry(self._p[i][j], nq_addr + i)
            for i in range(nq_data):
                self.circuit.cx(
                    compute_control(j, nq_addr, shift=i % self.nq_addr),
                    nq_addr + i
                )
        if self.keep_last_cx is False:
            self.circuit.data.pop(slice(-1, -self.nq_data-1, -1))
        if self.reverse_bits:
            self.circuit = self.circuit.reverse_bits()
        if self.measure:
            self.circuit.measure_all()
        if self.statevec:
            self.circuit.save_statevector()
        self._data = None
        self._max_val = None
        self._angles = None
        self._angles_qcrank = None

    def transpile(self, *args, **kwargs):
        self.circuit = transpile(self.circuit, *args, **kwargs)

    def bind_data(self, data):
        '''Enables binding the QCRANK circuit to data

        Args:
            data:
                Numerical data to bind to the parametrized QCRANK circuit:
                  * numpy array of size (2**nq_addr, nq_data)
                  * list of numpy arrays of size (2**nq_addr, nq_data)
                  * numpy array of size (2**nq_addr, nq_data, k)
        '''
        if not isinstance(data, (np.ndarray, list)):
            raise RuntimeError('data should be either numpy array or list of '
                               f'numpy array, got {isinstance(data)}')
        if isinstance(data, list):
            data = np.stack(data, axis=2)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data[..., np.newaxis]
        if data.shape[0] != 2**self.nq_addr:
            raise RuntimeError(
                f'Input data of incorrect shape {data.shape}, expecting '
                f'({2**self.nq_addr}, {self.nq_data}, ...) '
                '[(2**nq_addr, nq_data, k)]'
            )
        self._data = data
        self._color_bits = con_to_ang(data, self.nq_addr, self.nq_data)
        self._angles = rescale_bits_to_angle(self._color_bits)
        self._angles_qcrank = np.empty(self._angles.shape)
        for r in range(self._angles.shape[1]):
            self._angles_qcrank[:, r] = shifted_gray_permutation(
                sfwht(2 * self._angles[:, r]), r % self.nq_addr
            )

    @property
    def nq_addr(self):
        return self._nq_addr

    @property
    def nq_data(self):
        return self._nq_data

    @property
    def data(self):
        return self._data

    @property
    def angles_qcrank(self):
        return self._angles_qcrank

    def instantiate_circuits(self):
        '''Generates the instantiated circuits. '''
        if self.angles_qcrank is None:
            raise RuntimeError('Parametrized QCRANK circuit is not yet binded '
                               'to data. Run `bind_data` method first.`')
        circs = []
        for j in range(1):
            my_dict = {}
            for i in range(self.nq_data):
                my_dict[self.parameters[i]] = \
                    self.angles_qcrank[:, i]
            circ = self.circuit.bind_parameters(my_dict)
            circs.append(circ)
        return circs

    @property
    def parameters(self):
        '''Returns the parameter vectors.'''
        return self._p