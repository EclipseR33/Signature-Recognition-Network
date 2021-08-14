import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from dataset.dataset import HandWritingDataSet, get_train_df
from model.CNN.CNNmodel import ArcSRNet3, ArcSRNet4, ArcSRNet5, embedding
from model.KNN.KNNmodel import get_neighbors
from utils import plot_loss_lr
import warnings
import pickle


class CFG:
    # paths
    data_root = 'dataset/data/final'
    save_dir = 'log'
    save_notebook_path = os.path.join(save_dir, '1-1Arc3-111.pkl')

    # basic
    seed = 4039
    num_workers = 2
    image_size = 128
    epoch = 20
    batch_size = 24
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_notebook = True

    s = 30
    margin = 0.5

    lr = 1e-4

    model_params = {
        'num_block': [1, 1, 1, 1, 1],
        'groups': 4,
        'num_classes': -1
    }

    scheduler_params = {
        "lr_max": 1e-4 * batch_size,
        "lr_min": 1e-6,
        "lr_ramp_ep": 10,
        "lr_sus_ep": 0,
        "lr_decay": 0.8
    }

    # threshold
    threshold = 0.5
    get_threshold = True
    threshold_range = [0.01, 20, 0.01]
    n_neighbors = 50


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, lr, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                 last_epoch=-1):
        self.lr_start = lr
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]

        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) /
                  self.lr_ramp_ep * self.last_epoch +
                  self.lr_start)

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay **
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) +
                  self.lr_min)
        return lr


def train_image_model(dl, model, device, criterion, optimizer, scheduler, epoch):
    model.train()

    loss_score = AverageMeter()
    acc_score = AverageMeter()
    dl = tqdm(enumerate(dl), total=len(dl))

    for i, (img, label) in dl:
        batch_size = label.shape[0]

        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output, logits = model(img, label)

        loss = criterion(logits, label)
        loss.backward()

        pred = torch.argmax(logits, dim=1).to(device)
        count = torch.sum(pred == label.to(device)).item()
        acc_score.update(count / batch_size)

        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        dl.set_postfix(Avg_Loss=loss_score.avg, Loss=loss.detach().item(), Epoch=epoch,
                       LR=optimizer.param_groups[0]['lr'], Acc=acc_score.avg)
    if scheduler is not None:
        scheduler.step()

    return loss_score, acc_score


def eval_image_model(dl, model, device, criterion):
    model.eval()

    loss_score = AverageMeter()
    dl = tqdm(enumerate(dl), total=len(dl))
    with torch.no_grad():
        for i, (img, label) in dl:
            batch_size = label.shape[0]

            img = img.to(device)
            label = label.to(device)

            output, logits = model(img, label)

            loss = criterion(logits, label)

            loss_score.update(loss.detach().item(), batch_size)
            dl.set_postfix(Eval_Loss=loss_score.avg)
    return loss_score


class NoteBook:
    def __init__(self):
        self.train_lr = []
        self.train_loss = []
        self.train_acc = []
        self.f1_scores = {'x': [], 'y': []}
        self.model = {}

    def update_lr(self, lr):
        self.train_lr.append(lr)

    def update_loss(self, train_loss):
        self.train_loss.append(train_loss)

    def update_acc(self, train_acc):
        self.train_acc.append(train_acc)

    def update_f1_scores(self, x, y):
        self.f1_scores['x'].append(x)
        self.f1_scores['y'].append(y)

    def updata_model(self, key, value):
        self.model[key] = value


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    notebook = NoteBook()

    set_seed(CFG.seed)

    transform = transforms.Compose([
        transforms.Resize([CFG.image_size, CFG.image_size]),
        transforms.ToTensor()
    ])

    # !!!fold
    # fold = KFold(n_splits=5, shuffle=True, random_state=0)

    df = get_train_df(CFG.data_root)

    train_dataset = HandWritingDataSet(df, CFG.data_root, transform)

    train_dl = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers)

    if CFG.model_params['num_classes'] == -1:
        CFG.model_params['num_classes'] = train_dataset.classes

    model = ArcSRNet3(**CFG.model_params, device=CFG.device)
    model.to(CFG.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    scheduler = LRScheduler(optimizer, CFG.lr, **CFG.scheduler_params)

    notebook.updata_model('name', model.__class__.__name__)
    for k in CFG.model_params:
        notebook.updata_model(k, CFG.model_params[k])
    print(notebook.model)

    for epoch in range(CFG.epoch):
        notebook.update_lr(optimizer.state_dict()['param_groups'][0]['lr'])
        train_loss, train_acc = train_image_model(train_dl, model, CFG.device, criterion, optimizer, scheduler, epoch)
        notebook.update_loss(train_loss.avg)
        notebook.update_acc(train_acc.avg)

    # plt.figure()
    # plt = plot_loss_lr(notebook.train_loss, None, plt)
    # plt.show()
    # plt.figure()
    # plt = plot_loss_lr(notebook.train_lr, None, plt)
    # plt.show()

    embeddings = embedding(train_dl, model, device=CFG.device)
    _ = get_neighbors(df, embeddings, CFG.threshold,
                      get_threshold=CFG.get_threshold, threshold_range=CFG.threshold_range,
                      n_neighbors=min(CFG.n_neighbors, len(train_dataset)),
                      notebook=notebook)
    # print(notebook.train_lr)
    # print(notebook.f1_scores)
    print(_['pred_matches'])
    if CFG.save_notebook:
        with open(CFG.save_notebook_path, 'wb') as file:
            notebook_str = pickle.dumps(notebook)
            file.write(notebook_str)
