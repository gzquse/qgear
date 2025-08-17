#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from ._util import (
    compute_control,
    shifted_gray_permutation,
    sfwht,
    convert_shots_to_pdf,
    marginal_distribution,
    rescale_angles_to_fdata,
    rescale_data_to_angles,
    cnot_permutation
)


class _DecoderQCRANK(ABC):
    '''Abstract base class for QCRANK data decoders.'''
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
    def angles_to_idata(angles, max_val=256):
        return np.round(rescale_angles_to_fdata(angles, max_val=max_val))

    @staticmethod
    def angles_to_fdata(angles, max_val=256):
        return rescale_angles_to_fdata(angles, max_val=max_val)

    def dist_to_marginals(self, dist):
        out = np.empty((2**(self.nq_addr+1), self.nq_data, *dist.shape[1:]))
        for i in range(self.nq_data):
            t_out = [k + self.nq_addr for k in range(self.nq_data) if k != i]
            out[:, i] = marginal_distribution(dist, t_out)
        return out


class QKAtan2DecoderQCRANK(_DecoderQCRANK):
    '''Qiskit compatible atan2 decoder for QCRANK.'''
    def angles_from_yields(self, yields, is_numpy=False):
        '''Decodes the angles from the yields of a QCRANK experiment.'''
        if is_numpy:
            assert yields.ndim == 2  # expected shape:  [bitstrings,images]
            pdfs = yields
        else:  # yields is a Qiskit dictionary
            pdfs = self.yields_to_pdf(yields, self.nq_addr + self.nq_data)
        marginal_pdfs = self.dist_to_marginals(pdfs)
        marginal_pdfs = np.sqrt(marginal_pdfs)
        if self.keep_last_cx is False:
            for i in range(self.nq_data):
                marginal_pdfs[:, i] = cnot_permutation(
                    marginal_pdfs[:, i],
                    i % self.nq_addr,
                    self.nq_addr
                )
        return 2 * np.arctan2(marginal_pdfs[1::2], marginal_pdfs[::2])

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
        return 2 * np.arctan2(
            statevec[1::2], statevec[::2]
        )

    @staticmethod
    def yields_to_pdf(yields, nqubits, normalize=False):
        '''The yields are  expected as a list of dictionaries in the standard
        qiskit format. The number of qubits is the length of each bitstring.
        The normalize flag indicates if sum to 1 or not.'''
        if isinstance(yields, dict):
            yields = [yields]
        pdfs = np.empty([2**nqubits, len(yields)])
        for i, y in enumerate(yields):
            pdfs[:, i] = convert_shots_to_pdf(y, normalize=normalize)
        return pdfs


class ParametrizedQCRANK:
    '''An object to handle the parametrized QCRANK circuits.'''
    def __init__(self, nq_addr, nq_data,
                 decoder: Type[_DecoderQCRANK] = QKAtan2DecoderQCRANK,
                 keep_last_cx: bool = True,
                 measure: bool = True,
                 statevec: bool = False,
                 reverse_bits: bool = False,
                 barrier: bool = True,
                 parallel: bool = True):
        '''Initializes a parametrized QCRANK circuit with nq_addr qubits and
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
            barrier: bool (True)
                add a barrier to the circuit
            parallel: bool (True)
                execute the CNOTs in parallel or sequential

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
        self.parallel = parallel
        if statevec and measure:
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
        if barrier:
            self.circuit.barrier()
        # add nested and shifted uniform rotations
        for j in range(2**nq_addr):
            for i in range(nq_data):
                self.circuit.ry(self._p[i][j], nq_addr + i)
            for i in range(nq_data):
                self.circuit.cx(
                    compute_control(j, nq_addr,
                                    shift=i % self.nq_addr if parallel else 0
                                    ),
                    nq_addr + i
                )
        if self.keep_last_cx is False:
            self.circuit.data.pop(slice(-1, -self.nq_data-1, -1))
        if self.reverse_bits:
            self.circuit = self.circuit.reverse_bits()
        if measure:
            self.circuit.measure_all()
        if self.statevec:
            self.circuit.save_statevector()
        self._data = None
        self._max_val = None
        self._angles = None
        self._angles_qcrank = None

    def transpile(self, *args, **kwargs):
        self.circuit = transpile(self.circuit, *args, **kwargs)

    def bind_data(self, data, max_val):
        '''Enables binding the QCRANK circuit to data

        Args:
            data:
                Numerical data to bind to the parametrized QCRANK circuit:
                  * numpy array of size (2**nq_addr, nq_data)
                  * list of numpy arrays of size (2**nq_addr, nq_data)
                  * numpy array of size (2**nq_addr, nq_data, k)
            max_val:
                maximum value of the discrete data
        '''
        if not isinstance(data, (np.ndarray, list)):
            raise RuntimeError('data should be either numpy array or list of '
                               f'numpy array, got {isinstance(data)}')
        if isinstance(data, list):
            data = np.stack(data, axis=2)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data[..., np.newaxis]
        if data.shape[0] != 2**self.nq_addr or data.shape[1] != self.nq_data:
            raise RuntimeError(
                f'Input data of incorrect shape {data.shape}, expecting '
                f'({2**self.nq_addr}, {self.nq_data}, ...) '
                '[(2**nq_addr, nq_data, k)]'
            )
        self._data = data
        self._angles = rescale_data_to_angles(data, max_val)
        self._angles_qcrank = np.empty(self._angles.shape)
        for r in range(self._angles.shape[1]):
            self._angles_qcrank[:, r] = shifted_gray_permutation(
                sfwht(self._angles[:, r]),
                r % self.nq_addr if self.parallel else 0
            )
        self._max_val = max_val

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
    def max_val(self):
        return self._max_val

    @property
    def angles(self):
        return self._angles

    @property
    def angles_qcrank(self):
        return self._angles_qcrank

    def instantiate_circuits(self):
        '''Generates the instantiated circuits. '''
        if self.angles_qcrank is None:
            raise RuntimeError('Parametrized QCRANK circuit is not yet binded '
                               'to data. Run `bind_data` method first.`')
        circs = []
        for j in range(self.angles_qcrank.shape[2]):
            my_dict = {}
            for i in range(self.nq_data):
                my_dict[self.parameters[i]] = \
                    self.angles_qcrank[:, i, j]
            circ = self.circuit.assign_parameters(my_dict)
            circs.append(circ)
        return circs

    @property
    def parameters(self):
        '''Returns the parameter vectors.'''
        return self._p