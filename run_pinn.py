import os
import sys
import torch
import random
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from model.pinn import PINN_MLP
from data.dataset import BOXDataset, denormalize_field
from utils.utils import EMA
from utils.utils import save_gxbox_as_torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def train(model, loader, optimizer, writer, epoch, device):
    ema = EMA()
    model.train()
    for i, data in enumerate(tqdm.tqdm(loader)):
        coord, field, boundary_coord, boundary = data
        coord = coord.to(device, non_blocking=True)
        #field = field.cuda()
        boundary_coord = boundary_coord.to(device, non_blocking=True)
        boundary = boundary.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss_equation = model.loss_equation(coord)
        loss_boundary = model.loss_boundary(boundary_coord, boundary)
        loss = loss_equation + loss_boundary
        loss.backward()
        optimizer.step()
        ema.append(loss.item())

    writer.add_scalar("train_loss", ema.ema, global_step=epoch)

    return ema.ema


@torch.no_grad()
def evaluate(model, loader, writer, epoch, device):
    ema = EMA()
    model.eval()
    all_x, all_y, all_pred = [], [], []

    for i, data in enumerate(tqdm.tqdm(loader)):
        coord, field, boundary_coord, boundary = data
        boundary_coord = boundary_coord.to(device, non_blocking=True)
        boundary = boundary.to(device, non_blocking=True)
        # pred = model(coord)
        loss = model.loss_boundary(boundary_coord, boundary)
        # loss = F.mse_loss(pred, field)
        ema.append(loss.item())
        # all_pred.append(pred)
        # all_x.append(x)
        # all_y.append(y)

    # all_x = torch.cat(all_x, dim=0)
    # all_y = torch.cat(all_y, dim=0)
    # all_pred = torch.cat(all_pred, dim=0)

    #image = plot_1d(all_x, all_y, all_pred)

    if writer:
        writer.add_scalar("val_loss", ema.ema, global_step=epoch)
        # writer.add_figure("prediction", image, global_step=epoch)

    return ema.ema


@torch.no_grad()
def predict(model, loader, target_shape, device):
    model.eval()

    result = torch.empty(target_shape, dtype=torch.float, device=device)

    for i, data in enumerate(tqdm.tqdm(loader)):
        coord, field, boundary_coord, boundary = data
        assert len(coord) == 1
        pred = model(coord.to(device))
        f = denormalize_field(pred)[0]
        x, y, z = coord[0].long()
        result[:, z, y, x] = f

    return result


def main(
        experiment_name='potential_enlarge',
        data_file='output/b_field.pt',
        log_every=1,
        do_predict=False,
        checkpoint_path='output/potential_test/checkpoint_best.pth',
):
    torch.manual_seed(2023)
    random.seed(2023)
    np.random.seed(2023)

    model = PINN_MLP(last_relu=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not do_predict else "cpu")
    #device = torch.device("cpu")
    model = model.to(device)

    data_cube = torch.load(data_file).float()
    data_cube = data_cube[..., ::2, ::2, ::2]

    def normalize_data_cube(d):
        d_std, d_mean = torch.std_mean(d, dim=(-3, -2, -1), keepdim=True)
        return (d - d_mean) / d_std

    # DATA MEAN: [16.0314, 4.4297, 5.327] ## DATA STD: [92.8251, 93.0907, 141.2029]
    data_cube = normalize_data_cube(data_cube)

    val_ds = BOXDataset(data_cube)

    if do_predict:
        model.load_state_dict(torch.load(checkpoint_path))
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=1)
        result = predict(model, val_loader, data_cube.shape, device)
        result = result.detach().cpu()
        os.makedirs(f'output/{experiment_name}', exist_ok=True)
        torch.save(result, f'output/{experiment_name}/prediction.pt')
        exit(0)

    train_ds = BOXDataset(data_cube)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=512)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=512)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(f'output/{experiment_name}/logs')

    best = 1e16
    best_epoch = 0
    model_save_path = f'output/{experiment_name}/checkpoint_best.pth'

    for epoch in range(1000):
        train_loss = train(model, train_loader, optimizer, writer, epoch, device)
        val_loss = evaluate(model, val_loader, writer, epoch, device)

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print('Saving model to %s' % model_save_path)

        if (epoch % log_every) == 0:
            print('Epoch: %d train loss: %.4f val loss: %.4f best epoch: %d'
                  % (epoch, train_loss, val_loss, best_epoch))

    print('Training finished')
    model.load_state_dict(torch.load(model_save_path))
    print('Best checkpoint evaluation')
    val_loss = evaluate(model, val_loader, None, None, device)
    print('Best model val loss is %.4f' % val_loss)


if __name__ == '__main__':
    #from magnetic.utils import torch2vtk
    #torch2vtk(sys.argv[1])
    main()
    #save_gxbox_as_torch(sys.argv[1])
