#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from ._util import (
    compute_control,
    sfwht,
    gray_permutation,
    convert_shots_to_pdf,
    rescale_data_to_angles,
    rescale_angles_to_fdata,
    cnot_permutation
)


class ParametrizedFRQI:
    '''An object to handle the parametrized FRQI circuits.'''
    def __init__(self, nq_addr):
        '''Initializes a parametrized FRQI circuit with nq_addr qubits.

        Args:
            nq_addr:
                number of address qubits, total circuit is nq_addr + 1
        '''
        self.nq_addr = nq_addr

        self._p = ParameterVector('p', 2**nq_addr)
        self._circ = QuantumCircuit(nq_addr + 1)
        # add diffusion
        for i in range(nq_addr):
            self._circ.h(i)
        self._circ.barrier()
        # add uniform rotation
        for i, p_i in enumerate(self._p):
            self._circ.ry(p_i, nq_addr)
            self._circ.cx(compute_control(i, nq_addr), nq_addr)

    def __call__(self, data, max_val):
        '''Binds the parametrized FRQI to data and returns a FRQI circuit.

        Args:
            data:
                data compatible with parametrized FRQI circuit of nq_addr
                qubits. The supported options are:
                  * numpy array of length 2**nq_addr
                  * list of numpy arrays of length 2**nq_addr
                  * numpy array of size (2**nq_addr, k)
            max_val:
                maximum value of the discrete data
        '''
        return FRQI(self, data, max_val)

    @property
    def parameters(self):
        '''Returns the parameter vector.'''
        return self._p

    @property
    def circuit(self):
        '''Returns the parametrized circuit.'''
        return self._circ


class FRQI:
    '''An object to handle instantiated FRQI circuits.'''
    def __init__(self, parametrized_frqi: ParametrizedFRQI, data, max_val):
        '''
        Connects the parametrized circuits with discretized data of compatible
        dimension. The circuits by default include the last CX gate, do NOT
        reverse the order of bits (for little-endiannes correction), do NOT
        include measurements or state_vector checkpoints. These output options
        are configurable through `configure_output`.

        Args:
            parametrized_frqi (ParametrizedFRQI)
                A parametrized FRQI object on nq_addr address qubits.
            data
                Numerical data to bind to the parametrized FRQI circuit:
                  * numpy array of length 2**nq_addr
                  * list of numpy arrays of length 2**nq_addr
                  * numpy array of size (2**nq_addr, k)
            max_val:
                maximum value of the discrete data
        '''
        self._pfrqi = parametrized_frqi
        if not isinstance(data, (np.ndarray, list)):
            raise RuntimeError('data should be either numpy array or list of '
                               f'numpy array, got {isinstance(data)}')
        if isinstance(data, list):
            data = np.stack(data, axis=1)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = data[:, np.newaxis]
        self._data = data
        self._angles = rescale_data_to_angles(data, max_val)
        self._angles_frqi = gray_permutation(sfwht(self._angles))
        self._max_val = max_val
        self.keep_last_cx = None
        self.measure = None
        self.statevec = None
        self.reverse_bits = None

    def configure_output(self, keep_last_cx: bool, measure: bool,
                         statevec: bool, reverse_bits: bool):
        '''Configures the output format for the circuits. There are three
        user settings. These method returns a copy of the underlying object.

        Args:
            keep_last_cx: bool
                keep or discard final CX gate
            measure: bool
                adds measurements to all qubits
            statevec: bool
                set to true for state vector simulation
            reverse_bits: bool
                set to true for reversing the order of the bits

        !!! note
            `measure` and `statevec` cannot be simultaneously set to True. If
            both are input as True, only measure will be configured.

        !!! note
            without configuring the output circuits don't have measurements
        '''
        new = copy.deepcopy(self)
        new.keep_last_cx = keep_last_cx
        new.measure = measure
        if statevec and new.measure:
            raise RuntimeWarning('Ignoring statevec flag over measurement'
                                 'flag')
        else:
            new.statevec = statevec
        new.reverse_bits = reverse_bits
        return new

    def generate_circuits(self):
        circs = []
        for i in range(self.angles.shape[1]):
            circ = self._pfrqi.circuit.assign_parameters(
                    {self._pfrqi.parameters:
                     self.angles_frqi[:, i]}
                )
            if self.keep_last_cx is False:
                circ.data.pop()
            if self.reverse_bits:
                circ = circ.reverse_bits()
            if self.measure:
                circ.measure_all()
            if self.statevec:
                circ.save_statevector()
            circs.append(circ)
        return circs

    @property
    def nq_addr(self):
        return self._pfrqi.nq_addr

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
    def angles_frqi(self):
        return self._angles_frqi


class _DecoderFRQI(ABC):
    '''Abstract base class for FRQI data decoders.'''
    def __init__(self, frqi: FRQI) -> None:
        self._frqi = frqi

    @abstractmethod
    def angles_from_yields(self, yields):
        pass

    @abstractmethod
    def angles_from_statevec(self, statevec):
        pass

    @staticmethod
    def angles_to_data(angles, max_val=256):
        return np.round(rescale_angles_to_fdata(angles, max_val=max_val))


class QKAtan2DecoderFRQI(_DecoderFRQI):
    '''Qiskit compatible atan2 decoder for FRQI.'''
    def angles_from_yields(self, yields):
        '''The yields are  expected as a list of dictionaries in the standard
        qiskit format.'''
        pdfs = self.yields_to_pdf(yields, self._frqi.nq_addr + 1)
        pdfs = np.sqrt(pdfs)
        if self._frqi.keep_last_cx is False:
            pdfs = cnot_permutation(pdfs, 0, self._frqi.nq_addr)
        return 2 * np.arctan2(
            pdfs[1::2, ...], pdfs[::2, ...]
        )

    def angles_from_statevec(self, statevec):
        '''The statevector(s) are  expected as either option below:
            * numpy array of length 2**(nq_addr+1)
            * list of numpy arrays of length 2**(nq_addr+1)
            * numpy array of size (2**(nq_addr+1, k)
        '''
        if isinstance(statevec, list):
            statevec = np.stack(statevec, axis=1)
        statevec = np.abs(statevec)
        if self._frqi.keep_last_cx is False:
            statevec = cnot_permutation(statevec, 0, self._frqi.nq_addr)
        return 2 * np.arctan2(
            statevec[1::2, ...], statevec[::2, ...]
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