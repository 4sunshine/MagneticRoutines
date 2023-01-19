import time

import torch


def levi_civita_3d():
    e = torch.zeros((3, 3, 3), dtype=torch.float)
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[2, 1, 0] = e[0, 2, 1] = e[1, 0, 2] = -1
    return e


def curl_ein(bm):
    """
    bm - batched matrix B x Dim x Dim
    bm = [B, dF(x, y, z), dx(x, y, z)]
    ~2 times slower than curl
    """
    assert len(bm.shape) == 3
    lcv = levi_civita_3d().to(bm.device)
    return torch.einsum('bfx,ixf->bi', bm, lcv)


def curl(bm):
    result = torch.empty(bm.shape[:-2] + (3,), device=bm.device)
    result[..., 0] = bm[..., 2, 1] - bm[..., 1, 2]
    result[..., 1] = bm[..., 0, 2] - bm[..., 2, 0]
    result[..., 2] = bm[..., 1, 0] - bm[..., 0, 1]
    return result


def divergence(bm):
    return torch.sum(torch.diagonal(bm, dim1=-2, dim2=-1), dim=-1)


def test_curl():
    dummy = torch.tensor([[12., 3., 2.],
                          [12., 12., 4.],
                          [18., 9., 12.]], dtype=torch.float)
    dummy.unsqueeze_(0)
    curl_ = curl(dummy)
    assert torch.allclose(curl_, torch.tensor([[5., -16., 9.]], dtype=torch.float))
    print(curl_)
    div = divergence(dummy)
    print(div)

    count_loops = 100

    st = time.time()
    for _ in range(count_loops):
        z = curl_ein(dummy)

    et = time.time()
    print('Eincurl', (et - st) / count_loops)

    st = time.time()
    for _ in range(count_loops):
        z = curl(dummy)

    et = time.time()
    print('Rescurl', (et - st) / count_loops)

    assert torch.allclose(z, torch.tensor([[5., -16., 9.]], dtype=torch.float))


if __name__ == '__main__':
    test_curl()
