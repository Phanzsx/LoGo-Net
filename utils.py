import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pdb
from torchvision import transforms
from PIL import Image
import time
from seg_losses import *
from matrics import *


class Trainer(object):
    def __init__(self, model, optimizer, save_dir=None, save_freq=1):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _loop(self, data_loader, ep, is_train=True):
        tensor2img = transforms.ToPILImage()
        loop_loss_class, correct, loop_loss_seg, loop_iou, loop_dice = [], [], [], [], []
        mode = 'train' if is_train else 'test'
        for data, tar, label in tqdm(data_loader):
            # data, tar, label = data.cuda(), tar.cuda(), label.cuda()
            if is_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            out_class, out_seg = self.model(torch.cat([data, data, data], 1))
            n = out_seg.size(0)
            loss_class = F.cross_entropy(out_class, label)
            loss_seg = F.binary_cross_entropy(torch.sigmoid(out_seg.view(n, -1)), tar.view(n, -1)) + \
                       IOULoss(out_seg, tar)
            loss = 0.5 * loss_class + loss_seg

            loop_loss_class.append(loss_class.detach() / len(data_loader))
            loop_loss_seg.append(loss_seg.data / len(data_loader))
            out = (out_class.data.max(1)[1] == label.data).sum()
            correct.append(float(out) / len(data_loader.dataset))

            for j in range(n):
                loop_iou.append(iou_score(out_seg[j], tar[j]))
                loop_dice.append(dice_coef(out_seg[j], tar[j]))

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(mode + ': loss_class: {:.6f}, Acc: {:.6%}, loss_seg: {:.6f}, iou: {:.6f}, dice: {:.6f}'.format(
            sum(loop_loss_class), sum(correct), sum(loop_loss_seg), sum(loop_iou)/len(loop_iou), sum(loop_dice)/len(loop_dice)))
        return sum(loop_loss_class), sum(correct), sum(loop_loss_seg), sum(loop_iou)/len(loop_iou), sum(loop_dice)/len(loop_dice)

    def train(self, data_loader, ep):
        self.model.train()
        results = self._loop(data_loader, ep)
        return results

    def test(self, data_loader, ep):
        self.model.eval()
        results = self._loop(data_loader, ep, is_train=False)
        return results

    def loop(self, epochs, train_data, test_data, scheduler=None, save_freq=5):
        f = open(self.save_dir + 'log.txt', 'w')
        # f.write('train_loss_cls train_acc train_loss_seg train_iou train_dice ' + 
        #         'test_loss_cls test_acc test_loss_seg test_iou test_dice\n')
        f.close()
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print('epoch {}'.format(ep))
            train_results = np.array(self.train(train_data, ep))
            test_results = np.array(self.test(test_data, ep))
            with open(self.save_dir + 'log.txt', 'a') as f:
                for i in np.append(train_results, test_results):
                    f.write(str(round(i, 6)) + ' ')
                f.write('\n')
            if not ep % save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir:
            name = self.save_dir + 'train' + str(epoch) + 'models.pth'
            torch.save(self.model.state_dict(), name)
            # torch.save(self.model, name)