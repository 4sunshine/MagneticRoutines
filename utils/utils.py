import os

import torch
from magnetic.gx_box import GXBox


class EMA(object):
    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._counter = 0
        self._ema = 0

    @property
    def ema(self):
        return self._ema

    def append(self, value):
        if self._counter > 0:
            self._ema += self._alpha * (value - self._ema)
        else:
            self._ema = value
        self._counter += 1


def save_gxbox_as_torch(filename):
    box = GXBox(filename)
    b = box.field_to_torch(*box.field)
    j = box.field_to_torch(*box.curl)
    os.makedirs('output', exist_ok=True)
    torch.save(b, 'output/b_field.pt')
    torch.save(j, 'output/curl.pt')

