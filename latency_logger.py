# axonml_adapter/latency_logger.py
import torch
import numpy as np

class LatencyLogger:
    def __init__(self, threshold: float, dt: float, node_check=None):
        """
        Records the time at which the first AP occurs (Vm crosses threshold).

        :param threshold: activation threshold in mV
        :param dt: timestep in ms
        :param node_check: list of node indices to check (default: all)
        """
        self.threshold = threshold
        self.dt = dt
        self.node_check = node_check  # list of node indices
        self.reset()

    def reset(self):
        self.latency_ms = None

    def __call__(self, outputs):
        """
        Called by AxonML model after each simulation.

        :param outputs: dictionary with 'Vm': torch.Tensor of shape [T, F, 1, N]
        """
        vm = outputs['Vm']  # shape: [T, F, 1, N]
        vm_np = vm.cpu().numpy()[..., 0, :]  # shape: [T, F, N]

        for fiber_idx in range(vm_np.shape[1]):
            for t_idx in range(vm_np.shape[0]):
                nodes_to_check = self.node_check or range(vm_np.shape[2])
                for n_idx in nodes_to_check:
                    if vm_np[t_idx, fiber_idx, n_idx] >= self.threshold:
                        self.latency_ms = t_idx * self.dt
                        return  # stop at first AP

    def record(self):
        """Return the recorded latency value."""
        return self.latency_ms
