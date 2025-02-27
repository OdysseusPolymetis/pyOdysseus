# __init__.py (partie BFSUJASON)
"""
Bertalign initialization
"""
__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

from bertalign.encoder import Encoder

model_name = "LaBSE"
# On force GPU + plus gros batch
model = Encoder(
    model_name=model_name,
    device='cuda',
    batch_size=64,
    show_bar=False
)

from bertalign.aligner import Bertalign