from __future__ import print_function, absolute_import

import collections
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.meters import AverageMeter
from cdsm.models.triplet_loss import TripletLoss
from cdsm.models.triplet import SoftTripletLoss
from cdsm.models.crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from cdsm.utils.faiss_rerank import compute_jaccard_distance
from cdsm.models.Weighted_Contrastive_Loss import OSM_CAA_Loss

import numpy as np

class Trainer(object):
    def __init__(self, encoder, encoder_ema, memory=None, alpha=0.999):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.alpha = alpha
        self.num_cluster = None
        self.cluster = None

    def train(self, epoch, data_loader, optimizer, print_freq=20, train_iters=400,
              ce_soft_weight=0.5, num_instances=4):
        self.encoder.train()
        self.encoder_ema.train()
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_cluster).cuda()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_ema = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, inputs_ema, labels, indexes = self._parse_data(inputs)

            f_out = self.encoder(inputs)
            with torch.no_grad():
                f_out_ema = self.encoder_ema(inputs)

            loss = self.memory(f_out, f_out_ema, labels, indexes)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # mean model's parameters update with model
            self._update_ema_variables(self.encoder, self.encoder_ema, self.alpha, epoch*len(data_loader)+i)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_1 {:.3f} ({:.3f})\t'
                      'Loss_2 {:.3f}({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ema.val, losses_ema.avg))

    def _parse_data(self, inputs):
        imgs, imgs_e, pids, camids, indexes = inputs
        return imgs.cuda(), imgs_e.cuda(), pids.cuda(), indexes

    @torch.no_grad()
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha) # increasing function alpha
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data = ema_param.data * alpha + param.data * (1.0 - alpha)
