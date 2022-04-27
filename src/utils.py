import torch
import torch.nn.functional as F


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu

    elif activation == "gelu":
        return torch.nn.GELU

    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation)
        )
