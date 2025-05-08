# axonml_adapter/sfap_callback.py
import torch
import numpy as np

class SfapLogger:
    def __init__(self, threshold: float, dt: float, window_ms: tuple = (-1.0, 4.0)):
        """
        Extracts the spike waveform (sfAP) around the first suprathreshold point.

        :param threshold: activation threshold (mV)
        :param dt: timestep (ms)
        :param window_ms: time window (start, stop) relative to first AP (in ms)
        """
        self.threshold = threshold
        self.dt = dt
        self.window = window_ms
        self.sfap = None

    def __call__(self, outputs):
        """
        Called by AxonML after run(). Should extract the spike waveform.

        :param outputs: dict with 'Vm': [T x F x 1 x N]
        """
        vm = outputs['Vm']  # torch tensor
        vm_np = vm.cpu().numpy()[..., 0, :]  # shape: [T x F x N]

        for f in range(vm_np.shape[1]):
            for t in range(vm_np.shape[0]):
                if (vm_np[t, f, :] >= self.threshold).any():
                    t_start = int((t + int(self.window[0] / self.dt)))
                    t_end = int((t + int(self.window[1] / self.dt)))
                    t_start = max(0, t_start)
                    t_end = min(vm_np.shape[0], t_end)

                    self.sfap = vm_np[t_start:t_end, f, :]  # shape: [T_window x N]
                    return

    def record(self):
        """Return saved sfap waveform."""
        return self.sfap
