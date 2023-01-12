import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from magnetic.gx_box import GXBox
import functorch


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


def vmap_run():
    """https://pytorch.org/functorch/stable/generated/functorch.vmap.html"""
    def prod(x):
        return torch.prod(x, -1, keepdim=True)

    x = torch.tensor([[1., 2., 3.],
                      [1., 2., 3.],
                      [4., 5., 6.],
                      [1., 2., 3.]], requires_grad=True)

    batch_size, dim = x.shape[:2]
    gather_index = torch.arange(batch_size).repeat_interleave(dim).unsqueeze(1).unsqueeze(1).repeat_interleave(dim, -1)

    y = prod(x)
    y = y * x

    assert y.shape[1] == dim
    y = y.flatten()

    def c_jac():
        def get_vjp(v):
            return torch.autograd.grad(y, x, v)

        I_N = torch.eye(len(y))
        jacobian = functorch.vmap(get_vjp)(I_N)

        return jacobian

    jac = c_jac()[0]
    gathered_grad = torch.gather(jac, 1, gather_index)
    gathered_grad = gathered_grad.reshape((batch_size, dim, dim))
    # batch; d result x,y,z; d x,y,z.
    print(gathered_grad)



if __name__ == '__main__':
    # jacobian_test()
    vmap_run()
