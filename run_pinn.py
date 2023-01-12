import sys
import torch
import random
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from model.pinn import PINN_MLP
from data.dataset import BOXDataset
from utils.utils import EMA
from utils.utils import save_gxbox_as_torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def train(model, loader, optimizer, writer, epoch):
    ema = EMA()
    model.train()
    for i, data in enumerate(tqdm.tqdm(loader)):
        coord, field, boundary_coord, boundary = data
        coord = coord.cuda()
        #field = field.cuda()
        boundary_coord = boundary_coord.cuda()
        boundary = boundary.cuda()
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
def evaluate(model, loader, writer, epoch):
    ema = EMA()
    model.eval()
    all_x, all_y, all_pred = [], [], []

    for i, data in enumerate(loader):
        coord, field, boundary_coord, boundary = data
        pred = model(coord)
        loss = F.mse_loss(pred, field)
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


def main(experiment_name='potential_test'):
    model = PINN_MLP(last_relu=False).cuda()
    torch.manual_seed(2023)
    random.seed(2023)
    np.random.seed(2023)

    train_ds = BOXDataset('output/b_field.pt')
    val_ds = BOXDataset('output/b_field.pt')

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=1024)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=1024)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(f'output/{experiment_name}/logs')

    best = 1e16
    best_epoch = 0
    model_save_path = f'output/{experiment_name}/checkpoint_best.pth'

    for epoch in range(1000):
        train_loss = train(model, train_loader, optimizer, writer, epoch)
        val_loss = evaluate(model, val_loader, writer, epoch)

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            #torch.save(model.state_dict(), model_save_path)
            #print('Saving model to %s' % model_save_path)

        if (epoch % 10) == 0:
            print('Epoch: %d train loss: %.4f val loss: %.4f best epoch: %d'
                  % (epoch, train_loss, val_loss, best_epoch))

    print('Training finished')
    # model.load_state_dict(torch.load(model_save_path))
    print('Best checkpoint evaluation')
    val_loss = evaluate(model, val_loader, None, None)
    print('Best model val loss is %.4f' % val_loss)


if __name__ == '__main__':
    main()
    #save_gxbox_as_torch(sys.argv[1])
