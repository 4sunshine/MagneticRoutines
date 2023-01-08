import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


def lin_act(in_dim, out_dim, act, bias=True):
    if act:
        return nn.Sequential(nn.Linear(in_dim, out_dim, bias=bias), act)
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        #nn.init.normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0.)


def plot_1d(x, y, y_pred):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    xy = np.concatenate([x, y, y_pred], axis=1)
    x, y, y_pred = xy.T
    plt.clf()
    fig = plt.figure()
    plt.scatter(x, y)
    plt.scatter(x, y_pred)
    return fig


def fn_squared(x):
    return x ** 2


def identity(x):
    return x


class EquationDataset(Dataset):
    def __init__(self, x, target_fn=fn_squared, transform=identity):
        self.x = x.float()
        self.target_y = target_fn(self.x)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.transform(self.x[idx]).unsqueeze(0), self.target_y[idx].unsqueeze(0)


def create_datasets(num_points=80, val_part=0.3, transform=identity, target_fn=fn_squared):
    sampled_points = torch.linspace(-20, 20, num_points)
    shuffled_index = torch.randperm(num_points)
    num_val_points = int(val_part * num_points)
    train_points = sampled_points[shuffled_index[num_val_points:]]
    val_points = sampled_points[shuffled_index[:num_val_points]]
    train_dataset = EquationDataset(train_points, target_fn, transform=transform)
    val_dataset = EquationDataset(val_points, target_fn, transform=transform)
    return train_dataset, val_dataset


class Tracer(nn.Module):
    def __init__(self, last_relu=True):
        super(Tracer, self).__init__()

        in_out_feats = [(1, 100), (100, 10), (10, 1)]
        act_fn = [nn.ReLU(), nn.ReLU()]
        if last_relu:
            act_fn.append(nn.ReLU())
        else:
            act_fn.append(nn.Identity())

        self.net = nn.Sequential(*[
            lin_act(in_dim, out_dim, act) for (in_dim, out_dim), act in zip(in_out_feats, act_fn)
        ])

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

    def loss_boundary(self):
        pass

    def loss_equation(self, x):
        input_x = x.clone()
        input_x.requires_grad = True
        f = self(input_x)
        df_x = grad(f, input_x, grad_outputs=torch.ones_like(input_x), create_graph=True)[0]
        loss = F.mse_loss(df_x, 5 * torch.ones_like(df_x))
        return loss

    def loss_test(self, x, y):
        f = self(x)
        loss = F.mse_loss(f, y)
        # print(f.item(), (x**2).item(), loss.item())
        return loss

    def loss(self, x):
        pass


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


def train(model, loader, optimizer, writer, epoch):
    ema = EMA()
    model.train()
    for i, data in enumerate(loader):
        x, y = data
        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        ema.append(loss.item())

    writer.add_scalar("train_loss", ema.ema, global_step=epoch)

    return ema.ema


def train_diff(model, loader, optimizer, writer, epoch):
    ema = EMA()
    model.train()
    for i, data in enumerate(loader):
        x, y = data
        optimizer.zero_grad()
        loss = model.loss_equation(x)
        #y_pred = model(x)
        #loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        ema.append(loss.item())

    writer.add_scalar("diff_train_loss", ema.ema, global_step=epoch)

    return ema.ema


@torch.no_grad()
def evaluate(model, loader, writer, epoch):
    ema = EMA()
    model.eval()
    all_x, all_y, all_pred = [], [], []

    for i, data in enumerate(loader):
        x, y = data
        pred = model(x)
        loss = F.mse_loss(pred, y)
        ema.append(loss.item())
        all_pred.append(pred)
        all_x.append(x)
        all_y.append(y)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    image = plot_1d(all_x, all_y, all_pred)

    if writer:
        writer.add_scalar("val_loss", ema.ema, global_step=epoch)
        writer.add_figure("prediction", image, global_step=epoch)

    return ema.ema


def evaluate_diff(model, loader, writer, epoch):
    ema = EMA()
    model.eval()
    all_x, all_y, all_pred = [], [], []

    for i, data in enumerate(loader):
        x, y = data
        pred = model(x)
        loss = model.loss_equation(x)
        ema.append(loss.item())
        all_pred.append(pred)
        all_x.append(x)
        all_y.append(y)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    image = plot_1d(all_x, all_y, all_pred)

    if writer:
        writer.add_scalar("val_diff_loss", ema.ema, global_step=epoch)
        writer.add_figure("prediction_diff", image, global_step=epoch)

    return ema.ema


def test_run(experiment_name='test'):
    model = Tracer()
    torch.manual_seed(2023)
    random.seed(2023)
    np.random.seed(2023)

    train_ds, val_ds = create_datasets(num_points=80, val_part=0.3, transform=identity, target_fn=fn_squared)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=2)
    val_loader = DataLoader(val_ds, shuffle=False)
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
            torch.save(model.state_dict(), model_save_path)
            print('Saving model to %s' % model_save_path)

        if (epoch % 10) == 0:
            print('Epoch: %d train loss: %.4f val loss: %.4f best epoch: %d'
                  % (epoch, train_loss, val_loss, best_epoch))

    print('Training finished')
    model.load_state_dict(torch.load(model_save_path))
    print('Best checkpoint evaluation')
    val_loss = evaluate(model, val_loader, None, None)
    print('Best model val loss is %.4f' % val_loss)


def diff_run(experiment_name='diff', checkpoint_path='output/test/checkpoint_best.pth'):
    model = Tracer(last_relu=False)
    model.load_state_dict(torch.load(checkpoint_path))
    torch.manual_seed(2023)
    random.seed(2023)
    np.random.seed(2023)

    train_ds, val_ds = create_datasets(num_points=80, val_part=0.3, transform=identity, target_fn=fn_squared)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=2)
    val_loader = DataLoader(val_ds, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter(f'output/{experiment_name}/logs')

    best = 1e16
    best_epoch = 0
    model_save_path = f'output/{experiment_name}/checkpoint_best.pth'

    for epoch in range(1000):
        train_loss = train_diff(model, train_loader, optimizer, writer, epoch)
        val_loss = evaluate_diff(model, val_loader, writer, epoch)

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            # torch.save(model.state_dict(), model_save_path)
            # print('Saving model to %s' % model_save_path)

        if (epoch % 10) == 0:
            print('Epoch: %d train loss: %.4f val loss: %.4f best epoch: %d'
                  % (epoch, train_loss, val_loss, best_epoch))

    print('Training finished')
    # model.load_state_dict(torch.load(model_save_path))
    print('Best checkpoint evaluation')
    val_loss = evaluate_diff(model, val_loader, None, None)
    print('Best model val loss is %.4f' % val_loss)



if __name__ == "__main__":
    diff_run()
    #test_run()

