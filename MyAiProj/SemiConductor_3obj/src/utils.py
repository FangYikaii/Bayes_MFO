import numpy as np
import torch


def convert_to_list(data):
    if data is None:
        return None
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return data
    else:
        raise TypeError(f"Unsupported type: {type(data)}")
