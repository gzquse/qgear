#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import frqi
import numpy as np
import qiskit as qk


simulator = qk.Aer.get_backend('aer_simulator')


def test_end_to_end():
    nq_addr = 3
    param_frqi = frqi.ParametrizedFRQI(nq_addr)

    data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    max_val = 8
    shots = 200_000

    data_frqi = param_frqi(data, max_val=max_val)

    # state vector simulation with last CNOT
    data_frqi_sv_cx = data_frqi.configure_output(
        keep_last_cx=True,
        measure=False,
        statevec=True,
        reverse_bits=True
    )
    data_circs_sv_cx = data_frqi_sv_cx.generate_circuits()
    results = [simulator.run(c).result() for c in data_circs_sv_cx]
    svecs = [r.get_statevector(c) for r, c in zip(results, data_circs_sv_cx)]
    decoder = frqi.QKAtan2DecoderFRQI(data_frqi_sv_cx)
    angles_rec = decoder.angles_from_statevec(svecs)
    data_rec = np.ravel(decoder.angles_to_data(angles_rec, max_val=max_val))
    np.testing.assert_allclose(data_rec, data, rtol=1e-12)

    # state vector simulation w/o last CNOT
    data_frqi_sv_no_cx = data_frqi.configure_output(
        keep_last_cx=False,
        measure=False,
        statevec=True,
        reverse_bits=True
    )
    data_circs_sv_no_cx = data_frqi_sv_no_cx.generate_circuits()
    results = [simulator.run(c).result() for c in data_circs_sv_no_cx]
    svecs = [r.get_statevector(c) for r, c in
             zip(results, data_circs_sv_no_cx)]
    decoder = frqi.QKAtan2DecoderFRQI(data_frqi_sv_no_cx)
    angles_rec = decoder.angles_from_statevec(svecs)
    data_rec = np.ravel(decoder.angles_to_data(angles_rec, max_val=max_val))
    np.testing.assert_allclose(data_rec, data, rtol=1e-12)

    # yields decoder with last CNOT
    data_frqi_y_cx = data_frqi.configure_output(
        keep_last_cx=True,
        measure=True,
        statevec=False,
        reverse_bits=True
    )
    data_circs_y_cx = data_frqi_y_cx.generate_circuits()
    results = [simulator.run(c, shots=shots).result() for c in data_circs_y_cx]
    yields = [r.get_counts(c) for r, c in zip(results, data_circs_y_cx)]
    decoder = frqi.QKAtan2DecoderFRQI(data_frqi_y_cx)
    angles_rec = decoder.angles_from_yields(yields)
    data_rec = np.ravel(decoder.angles_to_data(angles_rec, max_val=max_val))
    np.testing.assert_allclose(data_rec, data, rtol=1e-12)

    # yields decoder w/o last CNOT
    data_frqi_y_no_cx = data_frqi.configure_output(
        keep_last_cx=False,
        measure=True,
        statevec=False,
        reverse_bits=True
    )
    data_circs_y_no_cx = data_frqi_y_no_cx.generate_circuits()
    results = [simulator.run(c, shots=shots).result() for c in
               data_circs_y_no_cx]
    yields = [r.get_counts(c) for r, c in zip(results, data_circs_y_no_cx)]
    decoder = frqi.QKAtan2DecoderFRQI(data_frqi_y_no_cx)
    angles_rec = decoder.angles_from_yields(yields)
    data_rec = np.ravel(decoder.angles_to_data(angles_rec, max_val=max_val))
    np.testing.assert_allclose(data_rec, data, rtol=1e-12)
