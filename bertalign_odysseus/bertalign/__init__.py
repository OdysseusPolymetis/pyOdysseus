"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.2.0"

import torch
from bertalign.encoder import Encoder

_DEFAULT_MODEL_NAME = "LaBSE"
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_SHOW_BAR = False

model = None
model_name = None

def set_model(model_name_or_path=None, device=None, batch_size=None, show_bar=None):
    global model, model_name

    chosen_model = model_name_or_path or _DEFAULT_MODEL_NAME
    chosen_device = device or _DEFAULT_DEVICE
    chosen_batch_size = batch_size if batch_size is not None else _DEFAULT_BATCH_SIZE
    chosen_show_bar = show_bar if show_bar is not None else _DEFAULT_SHOW_BAR

    model = Encoder(
        model_name_or_path=chosen_model,
        device=chosen_device,
        batch_size=chosen_batch_size,
        show_bar=chosen_show_bar,
    )
    model_name = chosen_model
    return model

def get_model():
    global model
    if model is None:
        set_model()
    return model

# initialise le modèle par défaut au chargement
set_model()

from bertalign.aligner import Bertalign
