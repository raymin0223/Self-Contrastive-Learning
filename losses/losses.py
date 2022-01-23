"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ContrastMemory
import numpy as np

eps = 1e-7


class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, selfcon_m_FG=False, selfcon_s_FG=False, supcon_s=False, mem=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] if not selfcon_m_FG else int(features.shape[0]/2)
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)    
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if not selfcon_m_FG and not selfcon_s_FG:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        elif selfcon_m_FG:
            contrast_count = int(features.shape[1] * 2)
            anchor_count = (features.shape[1]-1)*2
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        elif selfcon_s_FG:
            contrast_count = features.shape[1]
            anchor_count = features.shape[1]-1
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        
        # compute logits
        if mem is None:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature) #if self.represent else torch.matmul(anchor_feature, contrast_feature.T)
        else:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, torch.cat([contrast_feature.T, mem.queue.to(device)], dim=1)),
                self.temperature) #if self.represent else torch.matmul(anchor_feature, torch.cat([contrast_feature.T, mem.queue.to(device)], dim=1))
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        if mem is not None:
            mask = torch.cat([mask, torch.eq(labels.repeat(anchor_count, 1), mem.q_label.to(device)).float().to(device)], dim=1) 
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
            
        mask = mask * logits_mask
        if supcon_s:
            idx = mask.sum(1) != 0
            mask = mask[idx, :]
            logits_mask = logits_mask[idx, :]
            logits = logits[idx, :]
            batch_size = idx.sum()
            
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
            
        return loss


class KLLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=3.0):
        super(KLLoss, self).__init__()
        self.T = T

    def forward(self, logit_s, logit_t):
        p_s = F.log_softmax(logit_s/self.T, dim=1)
        p_t = F.softmax(logit_t.clone().detach()/self.T, dim=1)
        loss = -pow(self.T, 2)*(p_s * p_t).sum(dim=1).mean()
        
        return loss
    

class BYOLLoss(nn.Module):
    def forward(self, feats):
        loss = torch.tensor(0.0).cuda()
        
        f_t = feats[-1].clone().detach()
        for idx, f_s in enumerate(feats[0]):
            loss += (2 - 2 * (f_s * f_t).sum(dim=1)).mean()
            
        return loss
            
    
class CRDLoss(nn.Module):
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = _ContrastLoss(opt.n_data)
        self.criterion_s = _ContrastLoss(opt.n_data)
        self.temperature = opt.nce_t

    def forward(self, feats, idx, contrast_idx):
        loss = torch.tensor(0.0).cuda()
        
        f_t = feats[-1].clone().detach()
        for i, f_s in enumerate(feats[0]):
            update = True if i == len(feats[0])-1 else False
            out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx, update=update)
            loss += self.criterion_s(out_s)[0]
            loss += self.criterion_t(out_t)[0]
            
        return loss

    
class _ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(_ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
                           P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class MLCPCLoss(nn.Module):
    def __init__(self, opt):
        super(MLCPCLoss, self).__init__()
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = _MultiLabelLoss(opt.n_data, opt.cpc_alpha)
        self.criterion_s = _MultiLabelLoss(opt.n_data, opt.cpc_alpha)
        self.temperature = opt.nce_t

    def forward(self, feats, idx, contrast_idx):
        loss = torch.tensor(0.0).cuda()
        
        f_t = feats[-1].clone().detach()
        for i, f_s in enumerate(feats[0]):
            update = True if i == len(feats[0])-1 else False
            out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx, update=update)
            loss += self.criterion_s(out_s)
            loss += self.criterion_t(out_t)
            
        return loss
    
    
class _MultiLabelLoss(nn.Module):
    def __init__(self, n_data, alpha=1.0):
        super(_MultiLabelLoss, self).__init__()
        self.n_data = n_data
        self.alpha = alpha

    def forward(self, x):
        m = x.size(1) - 1

        alpha = self.alpha
        beta = (m + 1 - alpha) / m

        P_pos = x.select(1, 0)
        P_neg = x.narrow(1, 1, m)

        log_numerator = P_pos.log() 
        log_denominator = ((P_pos * alpha / m).mean() +
                           (P_neg * beta).mean() * m).log()

        loss = - log_numerator + log_denominator
        return loss.mean()
    

class CompReSS(nn.Module):
    def __init__(self, opt):
        super(CompReSS, self).__init__()
        self.criterion = _KLD()
        self.teacher_sample_similarities = _SampleSimilarities(opt.feat_dim, opt.nce_k, opt.nce_t)
#         self.student_sample_similarities = _SampleSimilarities(opt.feat_dim, opt.nce_k, opt.nce_t)
        self.temperature = opt.nce_t

    def forward(self, feats):
        loss = torch.tensor(0.0).cuda()
        
        teacher_feats = feats[-1].clone().detach()
        for idx, student_feats in enumerate(feats[0]):
            similarities_student = self.teacher_sample_similarities(student_feats, update=False)
            similarities_teacher = self.teacher_sample_similarities(teacher_feats, update=True)

            loss += self.criterion(similarities_teacher, similarities_student, self.temperature)
            
        return loss


class _KLD(nn.Module):
    def forward(self, targets, inputs, temp):
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')


class _SampleSimilarities(nn.Module):
    def __init__(self, feats_dim, queueSize, T):
        super(_SampleSimilarities, self).__init__()
        self.inputSize = feats_dim
        self.queueSize = queueSize
        self.T = T
        self.index = 0
        stdv = 1. / math.sqrt(feats_dim / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, feats_dim).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, feats_dim))

    def forward(self, q, update=True):
        batchSize = q.shape[0]
        queue = self.memory.clone()
        out = torch.mm(queue.detach(), q.transpose(1, 0))
        out = out.transpose(0, 1)
#         out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        if update:
            # update memory bank
            with torch.no_grad():
                out_ids = torch.arange(batchSize).cuda()
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, q)
                self.index = (self.index + batchSize) % self.queueSize

        return out
    
