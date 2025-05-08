# axonml_adapter/protocol.py
import numpy as np
import torch

def run_finite_amplitudes_protocol(model, ve, diam, dt, amps, callbacks):
    """
    Runs the AxonML model across a list of stimulation amplitudes and returns AP counts.

    :param model: AxonML surrogate model (e.g. SMF)
    :param ve: extracellular potential array [T, 1, 1, N]
    :param diam: fiber diameter [1]
    :param dt: timestep in ms
    :param amps: list of amplitudes to apply
    :param callbacks: list of callbacks to apply
    :return: Dictionary of amp -> callback records (e.g., AP counts or latency)
    """
    results = {}
    ve = np.float32(ve)
    diam = np.float32(diam)

    for amp in amps:
        amp_key = f"{amp:.3f}mA"
        print(f"Running amplitude {amp_key}...")
        amp_ve = ve * amp

        for cb in callbacks:
            if hasattr(cb, 'reset'):
                cb.reset()

        with torch.no_grad():
            model.run(amp_ve, diam, dt=dt, callbacks=callbacks, reinit=True)

        # Extract record() from callbacks
        result_entry = {}
        for cb in callbacks:
            if hasattr(cb, 'record'):
                record_val = cb.record()
                result_entry[cb.__class__.__name__] = record_val

        results[amp_key] = result_entry

    return results
