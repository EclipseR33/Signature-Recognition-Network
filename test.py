import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from dataset.dataset import HandWritingDataSet, TestDataSet, get_test_df, get_unique_threshold
from model.CNN.CNNmodel import Net3, Net4, Net5, SiameseSRNet, embedding
from model.KNN.KNNmodel import get_neighbors
from utils import plot_loss_lr
import random
import warnings


class CFG:
    # paths
    data_root = 'dataset/data/val'
    save_dir = 'log'

    # basic
    seed = 4039
    num_workers = 2
    image_size = 128
    epoch = 20
    batch_size = 24
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    s = 30
    margin = 0.5

    lr = 1e-3

    model_params = {
        'num_block': [1, 1, 1, 1, 1],
        'groups': 4,
        'num_classes': -1,
        'fc_dim': 256
    }

    scheduler_params = {
        "lr_max": 1e-3 * batch_size,
        "lr_min": 1e-4,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8
    }

    # threshold
    threshold = 0.01
    get_threshold = True
    threshold_range = [0.001, 0.1, 0.001]
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
    dl = tqdm(enumerate(dl), total=len(dl))
    for i, (img1, img2, label) in dl:
        batch_size = label.shape[0]

        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(img1, img2)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        dl.set_postfix(Avg_Loss=loss_score.avg, Loss=loss.detach().item(), Epoch=epoch,
                       LR=optimizer.param_groups[0]['lr'])
    if scheduler is not None:
        scheduler.step()
    return loss_score


def eval_image_model(dl, model, device):
    model.eval()

    acc_score = AverageMeter()
    dl = tqdm(enumerate(dl), total=len(dl))
    with torch.no_grad():
        for i, (img1, img2, label) in dl:
            batch_size = label.shape[0]

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output = model(img1, img2)

            pred = torch.argmax(output, dim=1).to(device)
            count = torch.sum(pred == label.to(device)).item()
            acc = count * 100 / batch_size

            acc_score.update(acc)
            dl.set_postfix(Eval_Acc=acc_score.avg)
    return acc_score


def read_image(image_path, transform, th=None):
    image = Image.open(image_path)
    image = image.convert('L')

    th, _ = get_unique_threshold(image, th)
    image = image.point(th, '1')
    image = transform(image)
    return image


def preprocess_image(image, transform):
    image = transform(image)
    image = image.unsqueeze_(1)
    image = image.to(CFG.device)
    return image


class NoteBook:
    def __init__(self):
        self.train_loss = []
        ####################################

    def update(self, train_loss):
        self.train_loss.append(train_loss)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    notebook = NoteBook()

    set_seed(CFG.seed)

    transform = transforms.Compose([
        transforms.Resize([CFG.image_size, CFG.image_size]),
        transforms.ToTensor()
    ])

    if CFG.model_params['num_classes'] == -1:
        CFG.model_params['num_classes'] = len(os.listdir(CFG.data_root))

    df = get_test_df(CFG.data_root)

    dataset = TestDataSet(df, CFG.data_root, transform)

    num_fold = 5
    for i_fold in range(num_fold):
        print(f'Fold: {i_fold}')

        df_train, df_test = df[df['fold'] != i_fold], df

        train_dataset = TestDataSet(df_train, CFG.data_root, transform)

        train_dl = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers)

        test_dataset = TestDataSet(df_test, CFG.data_root, transform)

        test_dl = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers)

        model = Net5(**CFG.model_params, device=CFG.device)
        test_model = SiameseSRNet(model, model.final_dim)
        test_model.to(CFG.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(test_model.parameters(), lr=CFG.lr)
        scheduler = LRScheduler(optimizer, CFG.lr, **CFG.scheduler_params)

        for epoch in range(CFG.epoch):
            train_loss = train_image_model(train_dl, test_model, CFG.device, criterion, optimizer, scheduler, epoch)
            train_acc = eval_image_model(train_dl, test_model, CFG.device)
            test_acc = eval_image_model(test_dl, test_model, CFG.device)

            notebook.update(train_loss.avg)

        df_test = df[df['fold'] == i_fold]
        max_label = df['label'].max()
        test_transform1 = transforms.Resize([CFG.image_size, CFG.image_size])
        test_transform2 = transforms.ToTensor()

        # plt.figure()
        # t_img = read_image(df['root'][0], test_transform1, th=220)
        # plt.imshow(t_img)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

        softmax = nn.Softmax()
        for i in range(int(max_label) + 1):
            t = np.where(df['label'] == str(i))[0]
            t2 = np.where(df['label'] != str(i))[0]
            t_df = df.iloc[t]
            t2_df = df.iloc[t2]
            choice = random.sample(range(len(t2_df)), len(t_df))
            t2_df = t2_df.iloc[choice]

            print(t_df)
            print(t2_df)
            # 同类
            for root1, root2, label1, label2 in zip(t_df['root'], t_df['root'], t_df['label'], t_df['label']):
                fig = plt.figure()
                ax = plt.subplot(1, 2, 1)
                img1 = read_image(root1, test_transform1)
                plt.imshow(img1)
                plt.xticks([])
                plt.yticks([])

                ax2 = plt.subplot(1, 2, 2)
                img2 = read_image(root2, test_transform1)
                plt.imshow(img2)
                plt.xticks([])
                plt.yticks([])

                img1 = preprocess_image(img1, test_transform2)
                img2 = preprocess_image(img2, test_transform2)

                output = test_model(img1, img2).to('cpu')
                pred = torch.argmax(output, dim=1)
                probability = softmax(output[0]).detach().numpy()
                print(output, probability, pred, label1, label2)

                plt.xlabel(f'{probability} {pred.item()}')
                plt.show()
            # 不同类
            for root1, root2, label1, label2 in zip(t_df['root'], t2_df['root'], t_df['label'], t2_df['label']):
                fig = plt.figure()
                ax = plt.subplot(1, 2, 1)
                img1 = read_image(root1, test_transform1)
                plt.imshow(img1)
                plt.xticks([])
                plt.yticks([])

                ax2 = plt.subplot(1, 2, 2)
                img2 = read_image(root2, test_transform1)
                plt.imshow(img2)
                plt.xticks([])
                plt.yticks([])

                img1 = preprocess_image(img1, test_transform2)
                img2 = preprocess_image(img2, test_transform2)

                output = test_model(img1, img2).to('cpu')
                pred = torch.argmax(output, dim=1)
                probability = softmax(output[0]).detach().numpy()
                print(output, probability, pred, label1, label2)

                plt.xlabel(f'{probability} {pred.item()}')
                plt.show()
            break
        break
