import sys
import copy

import pandas as pd
import torch
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional


def get_torch_device(device_name: Optional[str] = None) -> torch.device:
    """
    Retorna o dispositivo do PyTorch a ser utilizado.

    Args:
        device_name (Optional[str]): Nome do dispositivo (por exemplo, 'cuda' ou 'cpu').
            Se None, verifica se há CUDA disponível e utiliza o dispositivo apropriado.

    Returns:
        torch.device: O dispositivo configurado para execução (ex.: cuda:0 ou cpu).
    """
    print(torch.__version__)
    print(torch.version.cuda)
    if device_name is None:
        device_name = 'cpu'
        if torch.cuda.is_available():
            device_name = f'cuda:{torch.cuda.current_device()}'
    device = torch.device(device_name)
    return device
