import pickle
import os
import matplotlib.pyplot as plt

from train import NoteBook
from utils import *


def load(path):
    with open(path, 'rb') as file:
        notebook_str = file.read()
        notebook = pickle.loads(notebook_str)
    return notebook


def image1(plt, dir, paths):
    notebook = [load(os.path.join(dir, p)) for p in paths]
    # lr
    plt.figure()
    plt = plot_loss_lr(notebook[0].train_lr, None, plt, '', '#3bc9db')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.savefig('log/figure/lr.png')

    # loss
    plt.figure()
    plt = plot_loss_lr(notebook[0].train_loss, None, plt, 'A', '#ffa94d')
    plt = plot_loss_lr(notebook[1].train_loss, None, plt, 'B', '#69db7c')
    plt = plot_loss_lr(notebook[2].train_loss, None, plt, 'C', '#3bc9db')
    plt = plot_loss_lr(notebook[3].train_loss, None, plt, 'D', '#f783ac')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('log/figure/loss.png')

    # acc
    plt.figure()
    plt = plot_loss_lr(notebook[0].train_acc, None, plt, 'A', '#ffa94d')
    plt = plot_loss_lr(notebook[1].train_acc, None, plt, 'B', '#69db7c')
    plt = plot_loss_lr(notebook[2].train_acc, None, plt, 'C', '#3bc9db')
    plt = plot_loss_lr(notebook[3].train_acc, None, plt, 'D', '#f783ac')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('log/figure/acc.png')

    # f1-score
    plt.figure()
    plt = plot_f1_scores(notebook[0].f1_scores, plt, 'A', '#ffa94d')
    plt = plot_f1_scores(notebook[1].f1_scores, plt, 'B', '#69db7c')
    plt = plot_f1_scores(notebook[2].f1_scores, plt, 'C', '#3bc9db')
    plt = plot_f1_scores(notebook[3].f1_scores, plt, 'D', '#f783ac')
    plt.legend()
    plt.title('F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.savefig('log/figure/f1-score.png')

    # plt.figure()
    # plt = plot_loss_lr(notebook.train_acc, None, plt)
    # plt.show()
    # plt.figure()
    # plt = plot_f1_scores(notebook.f1_scores, plt)
    # plt.show()


if __name__ == '__main__':
    dir = 'log'

    paths = ['1-1Arc3-111.pkl', '1-2Arc4-1111.pkl', '1-3Arc5-11111.pkl', '1-4Arc5-12461.pkl']

    image1(plt, dir, paths)
