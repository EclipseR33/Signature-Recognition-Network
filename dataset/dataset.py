import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from tqdm import trange, tqdm
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def get_train_df(root):
    df = pd.DataFrame({'root': [], 'label': [], 'index': []})

    dirs = os.listdir(root)
    dirs = [int(d) for d in dirs]
    dirs.sort()
    dirs = dirs[:10]
    dirs = [str(d) for d in dirs]

    idx = 0
    for d in dirs:
        files = os.listdir(os.path.join(root, d))
        files = [os.path.join(root, d, f) for f in files]
        for f in files:
            df.loc[idx] = {'root': f, 'label': d, 'index': idx}
            idx += 1

    matches = df.groupby(['label'])['index'].unique().to_dict()
    df['matches'] = df['label'].map(matches)
    df['matches'] = df['matches'].apply(lambda x: ' '.join([str(int(i)) for i in x]))
    # print(df['matches'])
    # print(df['label'])
    return df


def get_test_df(root):
    df = pd.DataFrame({'root': [], 'label': [], 'index': [], 'fold': []})

    dirs = os.listdir(root)

    idx = 0
    for d in dirs:
        files = os.listdir(os.path.join(root, d))
        files = [os.path.join(root, d, f) for f in files]

        for i, f in enumerate(files):
            df.loc[idx] = {'root': f, 'label': d, 'index': idx, 'fold': i}
            idx += 1

    return df


def get_unique_threshold(image, th=None):
    try:
        image = np.array(image)

        scale = np.zeros(256)

        for i in image:
            for j in i:
                scale[j] += 1
        max_idx = np.where(scale == np.max(scale))[0][0]
        threshold = np.where(scale <= np.mean(scale))[0]
        threshold = np.where(threshold <= max_idx)[0][-1]
        # plt.figure()
        # plt.plot(scale)
        # plt.show()

    except:
        threshold = 120
    finally:
        if th is not None:
            threshold = th
        th = []
        for i in range(256):
            if i < threshold:
                th.append(0)
            else:
                th.append(1)

        return th, plt


def show_data(dataset, png_dir=None):
    for i in trange(len(dataset)):
        img, label = dataset[i]
        # img.show()

        if png_dir is not None:
            torchvision.utils.save_image(img, os.path.join(png_dir, f'{label.item()}_{i}.png'), nrow=5, normalize=True)


class HandWritingDataSet(Dataset):

    def __init__(self, df, root, transform):
        self.df = df
        self.transform = transform

        s = set(df['label'])
        self.classes = len(s)

        self.image_set = {'root': df['root'].to_list(), 'label': df['label'].to_list()}

        self.image_cache = []

        tq = tqdm(self.image_set['root'])
        for image_path in tq:
            image = Image.open(image_path)
            image = image.convert('L')

            th, _ = get_unique_threshold(image)
            image = image.point(th, '1')

            if self.transform is not None:
                image = self.transform(image)
            self.image_cache.append(image)

    def __len__(self):
        return len(self.image_set['root'])

    def __getitem__(self, idx):
        image = self.image_cache[idx]

        label = int(self.image_set['label'][idx])

        label = torch.tensor(label, dtype=torch.long)

        return image, label


class TestDataSet(Dataset):

    def __init__(self, df, root, transform):
        self.df = df
        self.transform = transform

        dirs = os.listdir(root)
        self.classes = len(dirs)

        self.image_set = {'root': df['root'].to_list(), 'label': df['label'].to_list()}

        self.image_cache = []

        tq = tqdm(self.image_set['root'])
        for image_path in tq:
            image = Image.open(image_path)
            image = image.convert('L')

            th, _ = get_unique_threshold(image)
            image = image.point(th, '1')

            if self.transform is not None:
                image = self.transform(image)
            self.image_cache.append(image)

        self.pair = []
        for i in range(len(self.image_set['root'])):
            for j in range(len(self.image_set['root'])):
                if i != j and self.image_set['label'][i] == self.image_set['label'][j]:
                    self.pair.append((i, j))
            cnt = 0
            while cnt < 4:
                j = np.random.choice(len(self.image_set['root']), 1)
                j = int(j)
                if i != j and self.image_set['label'][i] != self.image_set['label'][j]:
                    self.pair.append((i, j))
                    cnt += 1

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        idx1, idx2 = self.pair[idx]
        image1 = self.image_cache[idx1]
        image2 = self.image_cache[idx2]

        label1 = int(self.image_set['label'][idx1])
        label2 = int(self.image_set['label'][idx2])
        label = 1 if label1 == label2 else 0

        label = torch.tensor(label, dtype=torch.long)
        return image1, image2, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    df = get_train_df(r'data/val')
    print(df['matches'][:10])

    # transform = transforms.Compose([
    #     transforms.Resize([100, 100]),
    #     transforms.ToTensor()
    # ])
    # dataset = HandWritingDataSet(df, r'data/val', transform)
    # for i in range(len(dataset)):
    #     _, _1, label = dataset[i]
    #     print(label)
