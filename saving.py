# axonml_adapter/saving.py
import numpy as np
import os

def save_result_as_npz(output_dir: str, fiber_id: str, result_dict: dict):
    """
    Save structured simulation results into a .npz file for downstream analysis.

    :param output_dir: path to save directory
    :param fiber_id: string identifier for fiber, e.g. 'inner0_fiber3'
    :param result_dict: dictionary with results (Vm, threshold, latency, sfap, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{fiber_id}.npz')
    np.savez_compressed(output_path, **result_dict)
    print(f'[Saved] {output_path}')


# Example usage
if __name__ == '__main__':
    fiber_id = "inner0_fiber3"
    results = {
        "threshold": 1.23,
        "ap_latency": 0.44,
        "Vm": np.random.randn(600, 21),  # e.g. [T, N_nodes]
        "sfap": np.random.randn(100, 21)
    }
    save_result_as_npz("outputs", fiber_id, results)
