#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import qcrank
import numpy as np
import qiskit as qk

simulator = qk.Aer.get_backend('aer_simulator')


def test_end_to_end():
    nq_addr = 3
    nq_data = 2
    data = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
    data = data.T
    max_val = 16
    shots = 200_000

    # state vector simulation with last CNOTs
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=True,
        measure=False,
        statevec=True,
        reverse_bits=True
    )
    param_qcrank.bind_data(data, max_val)
    data_circs = param_qcrank.instantiate_circuits()
    results = [simulator.run(c).result() for c in data_circs]
    svecs = [r.get_statevector(c) for r, c in zip(results, data_circs)]
    angles_rec = param_qcrank.decoder.angles_from_statevec(svecs)
    data_rec = np.ravel(
        param_qcrank.decoder.angles_to_data(angles_rec, max_val=max_val)
    )
    np.testing.assert_allclose(data_rec, np.ravel(data), rtol=1e-12)

    # state vector simulation without last CNOTs
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=False,
        measure=False,
        statevec=True,
        reverse_bits=True
    )
    param_qcrank.bind_data(data, max_val)
    data_circs = param_qcrank.instantiate_circuits()
    results = [simulator.run(c).result() for c in data_circs]
    svecs = [r.get_statevector(c) for r, c in zip(results, data_circs)]
    angles_rec = param_qcrank.decoder.angles_from_statevec(svecs)
    data_rec = np.ravel(
        param_qcrank.decoder.angles_to_data(angles_rec, max_val=max_val)
    )
    np.testing.assert_allclose(data_rec, np.ravel(data), rtol=1e-12)

    # yields decoder with last CNOTs
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=True,
        measure=True,
        statevec=False,
        reverse_bits=True
    )
    param_qcrank.bind_data(data, max_val)
    data_circs = param_qcrank.instantiate_circuits()
    results = [simulator.run(c, shots=shots).result() for c in data_circs]
    yields = [r.get_counts(c) for r, c in zip(results, data_circs)]
    angles_rec = param_qcrank.decoder.angles_from_yields(yields)
    data_rec = np.ravel(
        param_qcrank.decoder.angles_to_data(angles_rec, max_val=max_val)
    )
    np.testing.assert_allclose(data_rec, np.ravel(data), rtol=1e-12)

    # yields decoder without last CNOTs
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=False,
        measure=True,
        statevec=False,
        reverse_bits=True
    )
    param_qcrank.bind_data(data, max_val)
    data_circs = param_qcrank.instantiate_circuits()
    results = [simulator.run(c, shots=shots).result() for c in data_circs]
    yields = [r.get_counts(c) for r, c in zip(results, data_circs)]
    angles_rec = param_qcrank.decoder.angles_from_yields(yields)
    data_rec = np.ravel(
        param_qcrank.decoder.angles_to_data(angles_rec, max_val=max_val)
    )
    np.testing.assert_allclose(data_rec, np.ravel(data), rtol=1e-12)


def test_18_bug():
    nq_addr = 1
    nq_data = 8
    n_addr = 2**nq_addr
    n_pix = nq_data*n_addr
    data = np.reshape(np.linspace(0, n_pix-1, n_pix), (nq_data, n_addr))
    data = data.T
    max_val = 32
    param_qcrank = qcrank.ParametrizedQCRANK(
        nq_addr,
        nq_data,
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=False,
        measure=False,
        statevec=True,
        reverse_bits=True
    )
    param_qcrank.bind_data(data, max_val)
    data_circs = param_qcrank.instantiate_circuits()
    results = [simulator.run(c).result() for c in data_circs]
    svecs = [r.get_statevector(c) for r, c in zip(results, data_circs)]
    angles_rec = param_qcrank.decoder.angles_from_statevec(svecs)
    data_rec = np.ravel(
        param_qcrank.decoder.angles_to_data(angles_rec, max_val=max_val)
    )
    np.testing.assert_allclose(data_rec, np.ravel(data), rtol=1e-12)