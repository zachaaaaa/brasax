# axonml_adapter/batch_runner.py
import numpy as np
import torch
from collections import defaultdict


def group_fibers_by_shape(fiber_list):
    """
    Group fibers that have the same node count and diameter into batches.

    :param fiber_list: list of fiber dicts with keys: 'id', 've', 'diam', etc.
    :return: dict {group_key: List[fiber_dict]}
    """
    groups = defaultdict(list)
    for f in fiber_list:
        key = (f['diam'], f['ve'].shape[-1])  # (diameter, node_count)
        groups[key].append(f)
    return groups


def run_fiber_batch(model, fiber_group, dt, callbacks):
    """
    Run AxonML model on a batch of fibers with identical shape.

    :param model: AxonML surrogate model
    :param fiber_group: list of fiber dicts with keys 'id', 've', 'diam'
    :param dt: timestep in ms
    :param callbacks: list of shared callbacks to use
    :return: dict {fiber_id: {callback_name: result}}
    """
    n_fibers = len(fiber_group)
    T, _, _, N = fiber_group[0]['ve'].shape

    ve_batch = np.stack([f['ve'] for f in fiber_group], axis=1)  # [T, F, 1, N]
    diam_batch = np.float32([f['diam'] for f in fiber_group])   # [F]

    for cb in callbacks:
        if hasattr(cb, 'reset'):
            cb.reset()

    with torch.no_grad():
        model.run(
            np.float32(ve_batch),
            diam_batch,
            dt=dt,
            callbacks=callbacks,
            reinit=True
        )

    results = {}
    for i, f in enumerate(fiber_group):
        record = {}
        for cb in callbacks:
            if hasattr(cb, 'record'):
                val = cb.record()
                # slice per-fiber if needed
                if isinstance(val, (np.ndarray, torch.Tensor)) and val.shape[0] == n_fibers:
                    record[cb.__class__.__name__] = val[i]
                else:
                    record[cb.__class__.__name__] = val
        results[f['id']] = record

    return results
