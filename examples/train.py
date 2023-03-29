# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random

import hdbscan
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cdsm import datasets
from cdsm import models
from cdsm.models.cm import DiversityMemory
from cdsm.trainers import Trainer, Trainer_cf
from cdsm.evaluators import Evaluator, extract_features
from cdsm.utils.data import IterLoader
from cdsm.utils.data import transforms as T
from cdsm.utils.data.sampler import RandomMultipleGallerySampler
from cdsm.utils.data.preprocessor import Preprocessor
from cdsm.utils.infomap_cluster import cluster_by_infomap, get_dist_nbr
from cdsm.utils.logging import Logger
from cdsm.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from cdsm.utils.faiss_rerank import compute_jaccard_distance
from cdsm.datasets.ContrastiveCrop import ContrastiveCrop

# import pandas as pd
import csv
start_epoch = best_mAP = best_mean_mAP = mAP = mean_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, mutual=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        # T.RandomApply([T.GaussianBlur([0.1, 2.0])], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    base_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.ToTensor(),
        normalizer,
        # T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, mutual=mutual,
                                base_transformer=base_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                              num_classes=0, pooling_type=args.pooling_type)

    # Load from checkpoint
    if args.resume:
        global start_epoch
        print('==> Load from checkpoint: initialize the model')
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model, strip='module.')
        start_epoch = checkpoint['epoch']

    if args.resume_ema:
        print('==> Load from checkpoint: initialize the mean model')
        checkpoint = load_checkpoint(args.resume_ema)
        copy_state_dict(checkpoint['state_dict'], model_ema, strip='module.')

    # use CUDA
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)

    for param in model_ema.parameters():
        param.detach_()

    return model, model_ema

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP, best_mean_mAP, mAP, mean_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # log epoch,num_clusters, num_outliers and mAP
    head = ['epoch', 'num_clusters', 'num_outliers', 'mAP', 'mean_mAP']
    csv_file = open(osp.join(args.logs_dir, 'log.csv'), 'w', newline='')
    writer = csv.DictWriter(csv_file, head)
    writer.writeheader()

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model, model_ema = create_model(args)

    # Evaluator
    evaluator_ema = Evaluator(model_ema)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer(model, model_ema)

    for epoch in range(start_epoch, args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model_ema, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            # DBSCAN dlucster
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == start_epoch:
                # DBSCAN cluster
                eps = args.eps
                eps_tight = eps - args.eps_gap
                eps_loose = eps + args.eps_gap
                print('Clustering criterion eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric="precomputed", n_jobs=-1)
                cluster_tight = DBSCAN(eps=eps_tight,min_samples=4,metric='precomputed',n_jobs=-1)
                cluster_loose = DBSCAN(eps=eps_loose,min_samples=4,metric='precomputed',n_jobs=-1)
                # cluster = hdbscan.HDBSCAN(metric='precomputed', prediction_data=True, core_dist_n_jobs=-1)
                trainer.cluster = cluster

            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_outlier = (pseudo_labels < 0).astype(int).sum()
            print("outliers/total = {}/{} = {:.3f}".format(num_outlier, len(pseudo_labels),
                                                           num_outlier / len(pseudo_labels)))
            print("core components: {}/{}".format(len(cluster.components_), len(pseudo_labels)))
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

            if args.memorybank == 'CMhybrid_v5':
                pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
                num_cluster_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
                num_outlier_tight = (pseudo_labels_tight < 0).astype(int).sum()
                pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
                num_cluster_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)
                num_outlier_loose = (pseudo_labels_loose < 0).astype(int).sum()
                print("outliers/total = {}/{} = {:.3f}".format(num_outlier_tight, len(pseudo_labels_tight),
                                                               num_outlier_tight / len(pseudo_labels_tight)))
                print("core components: {}/{}".format(len(cluster_tight.components_), len(pseudo_labels_tight)))
                print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster_tight))

                print("outliers/total = {}/{} = {:.3f}".format(num_outlier_loose, len(pseudo_labels_loose),
                                                               num_outlier_loose / len(pseudo_labels_loose)))
                print("core components: {}/{}".format(len(cluster_loose.components_), len(pseudo_labels_loose)))
                print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster_loose))

            print('lr: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

            core_sample_index = cluster.core_sample_indices_
            cluster.components_ = features[core_sample_index]
            del rerank_dist

            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num):
                labels = []
                outliers = 0
                for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                    if id != -1:
                        labels.append(id)
                    else:
                        labels.append(num + outliers)
                        outliers += 1
                return torch.Tensor(labels).long()


            ori_pseudo_labels = pseudo_labels.copy()
            if args.memorybank=='CMhybrid_v5':
                pseudo_labels = generate_pseudo_labels(pseudo_labels, num_cluster)
                pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_cluster_tight)
                pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_cluster_loose)

                # compute R_indep and R_comp
                N = pseudo_labels.size(0)
                label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()  # N*N
                label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
                label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

                # 1-something means that smaller R_comp indicates smaller intra-samples distances
                R_comp = torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
                R_indep = torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
                assert ((R_comp.min() >= 0) and (R_comp.max() <= 1))
                assert ((R_indep.min() >= 0) and (R_indep.max() <= 1))

                cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
                # cluster_img_num = collections.defaultdict(int)
                for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, ori_pseudo_labels)):
                    if label == -1: # remove un-clustered samples
                        continue
                    cluster_R_comp[label.item()].append(comp.item())
                    cluster_R_indep[label.item()].append(indep.item())
                    # cluster_img_num[label.item()] += 1

                cluster_R_comp = [max(cluster_R_comp[i])*0.2 for i in sorted(cluster_R_comp.keys())]
                cluster_R_indep = [max(cluster_R_indep[i])*0.2 for i in sorted(cluster_R_indep.keys())]

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[label].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_random_features(labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster*num_instances)
            for i in range(num_cluster):
                index = [i+k*num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(labels==i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features

        @torch.no_grad()
        def generate_muti_centers_features(labels, features, num_cluster, num_instances):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers_mean = [_ for _ in range(num_cluster * num_instances)]
            for idx in sorted(centers.keys()):
                temp = torch.chunk(torch.stack(centers[idx],dim=0), num_instances, dim=0)
                for i in range(num_instances):
                    if i+1 > len(temp):
                        centers_mean[idx + i*num_cluster] = torch.stack(centers[idx], dim=0).mean(0)
                    else:
                        centers_mean[idx + i*num_cluster] = temp[i].mean(0)

            memory_features = torch.stack(centers_mean, dim=0)
            return memory_features

        # Create memory bank
        memory = DiversityMemory(model.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                               num_instances=args.mem_instances, hard_weight=args.hard_weight).cuda()
        if args.memorybank=='CMhybrid':
            cluster_features = generate_cluster_features(pseudo_labels, features)
            memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        elif args.memorybank=='CMhybrid_v2':
            cluster_features = generate_cluster_features(pseudo_labels, features)
            memory_features = generate_random_features(pseudo_labels, features, num_cluster, args.num_instances)
            memory.features = F.normalize(torch.cat([cluster_features, memory_features], dim=0), dim=1).cuda()
        elif args.memorybank=='CMhybrid_v3':
            memory_features = generate_muti_centers_features(pseudo_labels, features, num_cluster, args.num_instances)
            memory.features = F.normalize(memory_features, dim=1).cuda()
        elif args.memorybank=='CMhybrid_v4':
            cluster_features = generate_cluster_features(pseudo_labels[cluster.core_sample_indices_],
                                                         features[cluster.core_sample_indices_])
            memory_features = generate_muti_centers_features(pseudo_labels[cluster.core_sample_indices_],
                                                         features[cluster.core_sample_indices_], num_cluster, args.mem_instances)
            memory.features = F.normalize(torch.cat([cluster_features, memory_features], dim=0), dim=1).cuda()
        elif args.memorybank=='CMhybrid_v5':
            cluster_features = generate_cluster_features(ori_pseudo_labels[cluster.core_sample_indices_],
                                                         features[cluster.core_sample_indices_])
            memory_features = generate_muti_centers_features(ori_pseudo_labels[cluster.core_sample_indices_],
                                                             features[cluster.core_sample_indices_], num_cluster, args.mem_instances)
            memory.features = F.normalize(torch.cat([cluster_features, memory_features], dim=0), dim=1).cuda()
            memory.r_comps_all = R_comp
            memory.r_indeps_all = R_indep
            memory.clu_r_comps = torch.tensor(cluster_R_comp)
            memory.clu_r_indeps = torch.tensor(cluster_R_indep)
            # memory.indep_thres = torch.tensor(indep_thres)
        else: # CM
            cluster_features = generate_cluster_features(pseudo_labels, features)

            memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), ori_pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        # Train Net
        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, mutual=True)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            # Mean model evaluation
            mean_mAP = evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mean_mAP > best_mean_mAP)
            best_mean_mAP = max(mean_mAP, best_mean_mAP)
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_mean_model.pth.tar'))

            print('\n * Finished epoch {:3d}  Mean model mAP: {:5.1%}  Mean model best: {:5.1%}{}\n'.
                  format(epoch, mean_mAP, best_mean_mAP, ' *' if is_best else ''))

            writer.writerow({head[0]: epoch, head[1]: num_cluster, head[2]: num_outlier, head[4]: mean_mAP})
            # writer.writerow({head[0]: epoch, head[1]: num_cluster, head[4]: mean_mAP})
        else:
            writer.writerow({head[0]: epoch, head[1]:num_cluster})
        csv_file.flush()
        lr_scheduler.step()

    csv_file.close()
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    # model_ema.load_state_dict(checkpoint['state_dict'])
    copy_state_dict(checkpoint['state_dict'], model_ema, strip='module.')
    evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hard-sample Guided Hybrid Contrast Learning for Unsupervised Person Re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--mem-instances', type=int, default=2)

    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('-rho', type=float, default=2e-3,
                        help='rho percentage, default=2e-3')

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")
    parser.add_argument('--hard-weight', type=float, default=0.5, help="hard weights")
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the memory bank")
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('-mb', '--memorybank', type=str, default='CM')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, metavar='PATH', default='')
    parser.add_argument('--resume-ema', type=str, metavar='PATH', default='')
    main()
