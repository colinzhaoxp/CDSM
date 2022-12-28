import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
# from .losses import CrossEntropyLabelSmooth, FocalTopLoss
from cdsm.models.triplet import SoftTripletLoss
from cdsm.models.crossentropy import CrossEntropyLabelSmooth

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None, None


def cm(inputs, inputs_ema, indexes, features, momentum=0.5):
    return CM.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None, None


def cm_hard(inputs, inputs_ema, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            # median = np.argpartition(np.array(distances), 3)[0]
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None, None, None


def cm_hybrid(inputs, inputs_ema, indexes, features, momentum=0.5):
    return CM_hybrid.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_hybrid_v2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//(ctx.num_instances + 1)
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        updated = set()
        for k, (instance_feature, index) in enumerate(zip(inputs, targets.tolist())):
            batch_centers[index].append(instance_feature)
            if index not in updated:
                indexes = [index + nums*i for i in range(1, (targets==index).sum()+1)]
                ctx.features[indexes] = inputs[targets==index]
                # ctx.features[indexes] = ctx.features[indexes] * ctx.momentum + (1 - ctx.momentum) * inputs[targets==index]
                # ctx.features[indexes] /= ctx.features[indexes].norm(dim=1, keepdim=True)
                updated.add(index)

        for index, features in batch_centers.items():
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None, None


def cm_hybrid_v2(inputs, inputs_ema, indexes, features, momentum=0.5, num_instances=16, *args):
    return CM_hybrid_v2.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)

class CM_hybrid_v3(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//ctx.num_instances
        # nums = len(ctx.features)//(ctx.num_instances + 1)
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        updated = set()
        for k, (instance_feature, index) in enumerate(zip(inputs, targets.tolist())):
            batch_centers[index].append(instance_feature)
            if index not in updated:
                indexes = [index + nums*i for i in range(0, (targets==index).sum())]
                # print(indexes)
                # ctx.features[indexes] = inputs[targets==index]
                ctx.features[indexes] = ctx.features[indexes] * ctx.momentum + (1 - ctx.momentum) * inputs[targets==index]
                ctx.features[indexes] /= ctx.features[indexes].norm(dim=1, keepdim=True)
                updated.add(index)

        # for index, features in batch_centers.items():
        #     mean = torch.stack(features, dim=0).mean(0)
        #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
        #     ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None, None

def cm_hybrid_v3(inputs, inputs_ema, indexes, features, momentum=0.5, num_instances=16, *args):
    return CM_hybrid_v3.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)

class CM_hybrid_v4(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//(ctx.num_instances+1)
        # nums = len(ctx.features)//(ctx.num_instances + 1)
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            features = torch.stack(features, dim=0)
            with torch.no_grad():
                dist = euclidean_dist(features, features)
                mean_dist = dist.mean(0)
                candidate = torch.argsort(mean_dist, descending=True)
                selected_index = torch.zeros(len(features), dtype=torch.bool)
                selected_index[candidate[0]] = True
                num_selected = 0
                for i in candidate:
                    if torch.ge(dist[i][selected_index], mean_dist[selected_index]).sum() == num_selected:
                        selected_index[i] = True
                        num_selected += 1
                        if num_selected == ctx.num_instances:
                            break
            # print(top_k_index)
            indexes = [index + nums * i for i in range(1, num_selected+1)]
            ctx.features[indexes] = ctx.features[indexes] * ctx.momentum + features[selected_index] * (1.0 - ctx.momentum)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1.0 - ctx.momentum) * features[torch.argmin(mean_dist)]

        return grad_inputs, None, None, None, None, None

def cm_hybrid_v4(inputs, inputs_ema, indexes, features, momentum=0.5, num_instances=16, *args):
    return CM_hybrid_v4.apply(inputs, inputs_ema, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)

class CM_hybrid_v5(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, inputs_ema, targets, features, clu_r_comps, bt_r_comps, clu_r_indeps, bt_r_indeps,
                momentum, num_instances):
        ctx.features = features
        ctx.clu_r_comps = clu_r_comps
        ctx.bt_r_comps = bt_r_comps
        ctx.clu_r_indeps = clu_r_indeps
        # ctx.indep_thres = indep_thres
        ctx.bt_r_indeps = bt_r_indeps
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs_ema, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//(ctx.num_instances+1)
        # nums = len(ctx.features)//(ctx.num_instances + 1)
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        # batch_r_comps_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)
            # batch_r_comps_centers[index].append(r_comp)

        for index, features in batch_centers.items():
            features = torch.stack(features, dim=0)
            r_comps = ctx.bt_r_comps[targets==index]
            r_indeps = ctx.bt_r_indeps[targets==index]
            with torch.no_grad():
                dist = euclidean_dist(features, features)
                mean_dist = dist.mean(0)
                candidate = torch.argsort(mean_dist, descending=True)
                selected_index = torch.zeros(len(features), dtype=torch.bool)
                selected_index[candidate[0]] = True
                num_selected = 1
                for i in candidate[1:]:
                    if torch.ge(dist[i][selected_index], mean_dist[selected_index]).sum() == num_selected:
                        if r_comps[i] >= ctx.clu_r_comps[index] or r_indeps[i] >= ctx.clu_r_indeps[index]:
                            selected_index[i] = True
                            num_selected += 1
                            if num_selected == ctx.num_instances:
                                break
            # print(top_k_index)
            indexes = [index + nums * i for i in range(1, num_selected+1)]
            ctx.features[indexes] = ctx.features[indexes] * ctx.momentum + features[selected_index] * (1.0 - ctx.momentum)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1.0 - ctx.momentum) * features[torch.argmin(mean_dist)]

        return grad_inputs, None, None, None, None, None, None, None, None, None, None

def cm_hybrid_v5(inputs, inputs_ema, indexes, features, clu_r_comps, bt_r_comps, clu_r_indeps, bt_r_indeps,
                 momentum=0.5, num_instances=16, *args):
    return CM_hybrid_v5.apply(inputs, inputs_ema, indexes, features, clu_r_comps, bt_r_comps, clu_r_indeps, bt_r_indeps,
                              torch.Tensor([momentum]).to(inputs.device), num_instances)

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class SoftInfoNCE(nn.Module):
    def __init__(self, W=0.5):
        super(SoftInfoNCE, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.W = W

    def forward(self, inputs, target_ema):
        topk = torch.topk(target_ema, 3, dim=1, largest=False, sorted=False)
        topk_softmax_values = F.softmax(topk.values, dim=1)
        soft_target = torch.scatter(torch.zeros_like(target_ema).cuda(), 1, topk.indices, topk_softmax_values)
        soft_loss = (- self.logsoftmax(inputs) * soft_target).mean(0).sum()
        return soft_loss

class DiversityMemory(nn.Module, ABC):
    
    __CMfactory = {
        'CM': cm,
        'CMhard':cm_hard,
    }

    def __init__(self, num_features, num_cluster, temp=0.05, momentum=0.2, mode='CM', hard_weight=0.2, smooth=0., num_instances=1):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_cluster = num_cluster

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_cluster, 0.1)
            print('>>> Using CrossEntropy with Label Smoothing.')
            # print('>>> Using Soft infoNCE loss')
            # self.cross_entropy = SoftInfoNCE()
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()

        if self.cm_type in ['CM', 'CMhard']:
            self.register_buffer('features', torch.zeros(num_cluster, num_features))
        elif self.cm_type=='CMhybrid':
            self.hard_weight = hard_weight
            self.register_buffer('features', torch.zeros(2*num_cluster, num_features))
        elif self.cm_type=='CMhybrid_v2':
            self.hard_weight=hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros((self.num_instances+1)*num_cluster, num_features))
        elif self.cm_type=='CMhybrid_v3':
            self.hard_weight = hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros(self.num_instances * num_cluster, num_features))
        elif self.cm_type=='CMhybrid_v4':
            self.hard_weight= hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros((self.num_instances+1) * num_cluster, num_features))
        elif self.cm_type=='CMhybrid_v5':
            self.hard_weight = hard_weight
            self.num_instances = num_instances
            self.register_buffer('features', torch.zeros((self.num_instances+1) * num_cluster, num_features))
            self.r_comps_all = None
            self.r_indeps_all = None
            self.clu_r_comps = None
            self.clu_r_indeps = None
            self.indep_thres = None
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))
        if self.cm_type in ['CMhybrid', 'CMhybrid_v2', 'CMhybrid_v3', 'CMhybrid_v4', 'CMhybrid_v5']:
            print('hard_weight: {}'.format(self.hard_weight))

    def forward(self, inputs, inputs_ema, targets, indexes):

        if self.cm_type in ['CM', 'CMhard']:
            # cos_loss = 2 - 2*torch.cosine_similarity(inputs, inputs_ema, dim=-1).mean()
            outputs = ClusterMemory.__CMfactory[self.cm_type](inputs, inputs_ema, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            # loss = self.cross_entropy(outputs, targets) + cos_loss
            return loss

        elif self.cm_type=='CMhybrid':
            outputs = cm_hybrid(inputs, inputs_ema, targets, self.features, self.momentum)

            outputs /= self.temp
            output_hard, output_mean = torch.chunk(outputs, 2, dim=1)
            loss = self.hard_weight * (self.cross_entropy(output_hard, targets) + (1 - self.hard_weight) * self.cross_entropy(output_mean, targets))
            return loss

        elif self.cm_type=='CMhybrid_v2':
            outputs = cm_hybrid_v2(inputs, inputs_ema, targets, self.features, self.momentum, self.num_instances)
            out_list = torch.chunk(outputs, self.num_instances+1, dim=1)
            out = torch.stack(out_list[1:], dim=0)
            neg = torch.max(out, dim=0)[0]
            pos = torch.min(out, dim=0)[0]
            mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
            logits = mask * pos + (1-mask) * neg
            loss = self.hard_weight * self.cross_entropy(out_list[0]/self.temp, targets) \
                + (1 - self.hard_weight) * self.cross_entropy(logits/self.temp, targets)
            return loss

        elif self.cm_type=='CMhybrid_v3':
            outputs = cm_hybrid_v3(inputs, inputs_ema, targets, self.features, self.momentum, self.num_instances)
            out_list = torch.chunk(outputs, self.num_instances, dim=1)
            loss = 0
            for out in out_list:
                loss += self.cross_entropy(out/self.temp, targets)
            # loss = loss / self.num_instances
            return loss

        elif self.cm_type=='CMhybrid_v4':
            outputs = cm_hybrid_v4(inputs, inputs_ema, targets, self.features, self.momentum, self.num_instances)
            out_list = torch.chunk(outputs, self.num_instances+1, dim=1)
            out = torch.stack(out_list[1:], dim=0)
            neg = torch.max(out,dim=0)[0]
            pos = torch.mean(out,dim=0)
            mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
            logits = mask * pos + (1-mask) * neg
            loss = self.cross_entropy(out_list[0]/self.temp, targets) * (1-self.hard_weight) + \
                    self.cross_entropy(logits/self.temp, targets) * self.hard_weight
            return loss

        elif self.cm_type=='CMhybrid_v5':
            bt_r_comps = self.r_comps_all[indexes]
            bt_r_indeps = self.r_indeps_all[indexes]
            outputs = cm_hybrid_v5(inputs, inputs_ema, targets, self.features, self.clu_r_comps, bt_r_comps, self.clu_r_indeps, bt_r_indeps, self.momentum, self.num_instances)
            out_list = torch.chunk(outputs, self.num_instances + 1, dim=1)
            out = torch.stack(out_list[1:], dim=0)
            neg = torch.max(out, dim=0)[0]
            pos = torch.mean(out, dim=0)
            mask = torch.zeros_like(out_list[0]).scatter_(1, targets.unsqueeze(1), 1)
            logits = mask * pos + (1 - mask) * neg
            loss = self.cross_entropy(out_list[0] / self.temp, targets) * (1 - self.hard_weight) + \
                   self.cross_entropy(logits / self.temp, targets) * self.hard_weight
            return loss

