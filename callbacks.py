# axonml_adapter/callbacks.py
from axonml_adapter.latency_logger import LatencyLogger
from axonml.models.callbacks import VmLogger, APCount


def translate_saving_to_callbacks(saving_config: dict, threshold: float, dt: float, n_nodes: int):
    """
    Translate ASCENT sim.json["saving"] config to a list of AxonML-compatible callbacks.
    
    :param saving_config: dict from sim.json
    :param threshold: activation threshold value (mV)
    :param dt: timestep (ms)
    :param n_nodes: number of nodes in the fiber
    :return: list of callbacks
    """
    callbacks = []

    # Always include APCount for threshold or activation evaluation
    callbacks.append(APCount(node_check=list(range(n_nodes)), threshold=threshold, dt=dt))

    # Optional: save full Vm
    if saving_config.get("Vm"):
        callbacks.append(VmLogger())

    # Optional: log latency of AP trigger
    if saving_config.get("aplatency"):
        callbacks.append(LatencyLogger(threshold=threshold, dt=dt))

    # Optional: add sfap logger (to be implemented in sfap_callback.py)
    if saving_config.get("sfap"):
        from axonml_adapter.sfap_callback import SfapLogger
        callbacks.append(SfapLogger(threshold=threshold, dt=dt))

    return callbacks
