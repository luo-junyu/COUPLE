import argparse
import torch
import torch.nn as nn
from model import *
from load_data import *
from utils import *
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import faiss
import networkx as nx
import os
import logging
from evaluate import mean_average_precision
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('log.txt', 'a'))


seed_setting(seed=2001)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Office-Home', choices=['Office-Home', 'Office-31', 'Digits'])
    parser.add_argument('--nbit', type=int, default=64, choices=[16, 32, 64, 128])
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=70)
    parser.add_argument('--inter', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--w', type=float, default=8e-6)
    parser.add_argument('--lamda1', type=float, default=1)
    parser.add_argument('--lamda2', type=float, default=1)
    parser.add_argument('--lamda3', type=float, default=100)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--ratio2', type=float, default=0.5)
    parser.add_argument('--ratio3', type=float, default=0.9)
    parser.add_argument('--domain', type=str, default='ArtToReal_World') 

    args = parser.parse_args()
    print(args)

    return args

def train(args, source_loader, target_train_loader, target_test_loader, n_class, dim1, source_dataset, target_dataset):
    model = DDPA(args, n_class, dim1)
    model.cuda()
    criterion_l2 = nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.w)
    model.train()

    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    for epoch in range(1, args.num_epoch + 1):
        correct = 0
        total_tgt = 0

        mnn = MNN(k=3)
        edge_index = mnn.build_mnn_graph(source_dataset.images, source_dataset.labels, target_dataset.images)
        mnn_graph = nx.Graph()
        mnn_graph.add_edges_from(edge_index.t().tolist())
        pgs = nx.pagerank(mnn_graph)
        len_source = len(source_dataset.labels)
        len_target = len(target_dataset.labels)

        target_pgs = torch.zeros(len_target)

        for i in range(len_target):
            tgt_index = i + len_source
            if tgt_index in pgs:
                target_pgs[i] = pgs[tgt_index]

        topk = int(0.5 * len_target)
        _, conf_indices = torch.topk(target_pgs, topk)

        for _source_data, _target_data in zip(source_loader, target_train_loader):
            optimizer.zero_grad()
            data_source, label_source, index_source = _source_data
            data_target, label_target, index_target = _target_data
            data_source = data_source.cuda()
            data_target = data_target.cuda()
            label_source = label_source.cuda()
            label_target = label_target.cuda()
            source_feat, source_h, source_clf, target_feat, target_h, target_clf = model(data_source, data_target)
            source_clf_loss = criterion(source_clf, label_source.squeeze())
            target_plabel = target_clf.argmax(dim=1).detach()
            sub_conf_indices = []
            for i, idx in enumerate(index_target):
                if i in sub_conf_indices:
                    sub_conf_indices.append(i)
            sub_conf_indices = torch.tensor(sub_conf_indices).cuda()
            mask = torch.zeros(len(target_plabel)).cuda()
            mask[sub_conf_indices] = 1
            target_feat_p = target_clf * mask.unsqueeze(1).float()
            target_clf_loss = criterion(target_feat_p, target_plabel)
            d = source_feat.size(1)
            source_len = source_feat.size(0)
            index = faiss.IndexFlatL2(d)
            source_feat_ = source_feat.clone().detach()
            target_feat_ = target_feat.clone().detach()
            index.add(source_feat_.contiguous().cpu().numpy())
            D, I = index.search(target_feat_.cpu().numpy(), 1)
            low_level_mixup_feats = []
            low_level_mixup_labels = []
            high_level_mixup_feats = []
            high_level_mixup_labels = []
            for i, neighbors in enumerate(I):
                for j, n in enumerate(neighbors):
                    if i >= len(label_source) or n >= len(target_plabel):
                        continue
                    if label_source[i] == target_plabel[n]:
                        low_level_mixup_feats.append(mixup(data_source[i], data_target[n], 0.8))
                        low_level_mixup_labels.append(label_source[i])
                    elif n in sub_conf_indices:
                        high_level_mixup_feats.append(mixup(source_feat[i], target_feat[n], 0.8))
                        source_one_hot = torch.zeros(n_class)
                        source_one_hot[label_source[i]] = 1
                        target_one_hot = torch.zeros(n_class)
                        target_one_hot[target_plabel[n]] = 1
                        soft_label = 0.8 * source_one_hot + 0.2 * target_one_hot
                        high_level_mixup_labels.append(soft_label)
            low_level_mixup_loss = 0
            if len(low_level_mixup_feats) > 1:
                low_level_mixup_feats = torch.stack(low_level_mixup_feats).detach()
                low_level_mixup_labels = torch.stack(low_level_mixup_labels)
                low_level_mixup_feats = low_level_mixup_feats.cuda().detach()
                low_level_mixup_labels = low_level_mixup_labels.cuda().detach()
                low_level_mixup_clf = model.forward_clf(low_level_mixup_feats)
                low_level_mixup_loss = criterion(low_level_mixup_clf, low_level_mixup_labels.squeeze())
            high_level_mixup_loss = 0
            if len(high_level_mixup_feats) > 1:
                high_level_mixup_feats = torch.stack(high_level_mixup_feats)
                high_level_mixup_labels = torch.stack(high_level_mixup_labels)
                high_level_mixup_feats = high_level_mixup_feats.cuda().detach()
                high_level_mixup_labels = high_level_mixup_labels.cuda().detach()
                high_level_mixup_clf = model.classify(high_level_mixup_feats)
                high_level_mixup_loss = criterion_l2(high_level_mixup_clf, high_level_mixup_labels)
            correct_sample = torch.sum(target_plabel.unsqueeze(1) == label_target)
            n_sample = label_target.size(0)

            correct += correct_sample
            total_tgt = total_tgt + n_sample
            source_b = torch.sign(source_h)
            target_b = torch.sign(target_h)
            sign_loss = args.ratio2 * criterion_l2(source_h, source_b) + (1 - args.ratio2) * criterion_l2(target_h, target_b)

            label_source_onehot = torch.eye(n_class).cuda()[label_source.squeeze(1), :]
            S_I = label_source_onehot.mm(label_source_onehot.t())
            S_I = S_I.cuda()

            h_norm_s = F.normalize(source_h)
            S_h_s = h_norm_s.mm(h_norm_s.t())
            S_h_s[S_h_s < 0] = 0

            relation_recons_loss1 = criterion_l2(S_h_s, 1.1 * S_I)
            F_S = F.normalize(source_feat)
            F_T = F.normalize(target_feat)
            S_T_feat = F_S.mm(F_T.t())
            h_norm_s = F.normalize(source_h)
            h_norm_t = F.normalize(target_h)
            S_T_h = h_norm_s.mm(h_norm_t.t())
            S_T_h[S_T_h < 0] = 0

            relation_recons_loss2 = criterion_l2(S_T_feat, S_T_h)
            relation_recons_loss = args.ratio3 * relation_recons_loss1 + (1 - args.ratio3) * relation_recons_loss2
            sup_loss = (source_clf_loss + target_clf_loss) * args.lamda1
            sign_loss = sign_loss * args.lamda2
            mixup_loss = (low_level_mixup_loss*0.1 + high_level_mixup_loss) * args.lamda1
            recon_loss = relation_recons_loss * args.lamda3
            loss =  sup_loss + recon_loss + sign_loss + mixup_loss
            loss.backward()
            optimizer.step()


        acc = 100. * correct / total_tgt
        print(f'Epoch [{epoch}/{args.num_epoch}]: Accuracy: {acc:.2f}%')

        save_model(f'models/{args.dataset}/{args.domain}/{args.nbit}/{map}.pth', model)

class MNN(object):
    def __init__(self, k=10):
        self.k = k
    
    def build_mnn_graph(self, src_features, src_labels, tgt_features):
        src_sim = torch.mm(src_features, src_features.t())
        src_sim = src_sim / src_sim.norm(dim=1, keepdim=True)
        src_labels = src_labels.argmax(dim=1)
        src_label_mask = src_labels.unsqueeze(1) == src_labels.unsqueeze(0)
        src_sim[~src_label_mask] = -1
        cross_sim = torch.mm(src_features, tgt_features.t())
        cross_sim = cross_sim / cross_sim.norm(dim=1, keepdim=True)
        tgt_sim = torch.mm(tgt_features, tgt_features.t())
        tgt_sim = tgt_sim / tgt_sim.norm(dim=1, keepdim=True)
        
        src_indices = torch.topk(src_sim, k=self.k, dim=-1)[1]
        src_edge_index = torch.stack([src_indices.view(-1), torch.arange(src_indices.size(0)).repeat_interleave(self.k).to(src_indices.device)])
        
        d = src_features.size(1)
        source_len = src_features.size(0)
        index = faiss.IndexFlatL2(d)
        index.add(src_features.contiguous().cpu().numpy())
        D, I = index.search(tgt_features.cpu().numpy(), self.k)

        cross_edge_indexs = []
        for i, neighbors in enumerate(I):
            for n in neighbors:
                cross_edge_indexs.append([i, n + source_len])

        cross_edge_index = torch.tensor(cross_edge_indexs).t()
        edge_index = torch.cat([src_edge_index, cross_edge_index], dim=1)
        
        return edge_index

def mixup(x1, x2, alpha=0.2):
    mixed_x = alpha * x1 + (1 - alpha) * x2
    return mixed_x

def save_model(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def test(model, query_loader, retrieval_loader):
    model.eval().cuda()
    re_BI, re_L, qu_BI, qu_L = compress(retrieval_loader, query_loader, model)

    re_BI = re_BI.cuda()
    re_L = re_L.cuda()
    qu_BI = qu_BI.cuda()
    qu_L = qu_L.cuda()
    mAP = mean_average_precision(re_BI, qu_BI, re_L, qu_L, 'cuda') * 100
    return mAP

def code_save(model, database_loader, query_loader):

    model.eval().cuda()
    re_BI, re_L, qu_BI, qu_L = compress(database_loader, query_loader, model)

    _dict = {
        'db_code': re_BI.cpu().numpy(),
        'db_label':re_L.cpu().numpy(),
        'qu_code': qu_BI.cpu().numpy(),
        'qu_label':qu_L.cpu().numpy(),
    }

    save_path = f'hashcode/{args.dataset}/{args.domain}_{args.nbit}.mat'
    if not os.path.exists(f'hashcode/{args.dataset}'):
        os.makedirs(f'hashcode/{args.dataset}')
    sio.savemat(save_path, _dict)

    return 0

def compress(database_loader, query_loader, model):
    re_BI = []
    re_L = []
    for _, (data_I, data_L, _) in enumerate(database_loader):
        if data_L.shape[0] <= 1:
            continue
        with torch.no_grad():
            var_data_I = data_I.cuda()
            code_I = model.predict(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I)
        data_L = data_L.squeeze()
        data_L_onehot = torch.zeros(data_L.size(0), model.n_class)
        data_L_onehot.scatter_(1, data_L.unsqueeze(1), 1)
        re_L.extend(data_L_onehot)
    qu_BI = []
    qu_L = []
    for _, (data_I, data_L, _) in enumerate(query_loader):
        if data_L.shape[0] <= 1:
            continue
        with torch.no_grad():
            var_data_I = data_I.cuda()
            code_I = model.predict(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I)
        data_L = data_L.squeeze()
        data_L_onehot = torch.zeros(data_L.size(0), model.n_class)
        data_L_onehot.scatter_(1, data_L.unsqueeze(1), 1)
        qu_L.extend(data_L_onehot)
    re_BI = torch.stack(re_BI)
    re_L = torch.stack(re_L)
    qu_BI = torch.stack(qu_BI)
    qu_L = torch.stack(qu_L)

    return re_BI, re_L, qu_BI, qu_L

if __name__ == '__main__':
    args = get_args()
    source_domain = args.domain.split('To')[0]
    target_domain = args.domain.split('To')[1]

    print('source domain: ' + source_domain)
    print('target domain: ' + target_domain)

    if args.dataset == 'Office-Home': 
        base_path = 'datasets/OfficeHome_mat/'
        source_loader, n_class, dim1, source_dataset = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader, target_dataset = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Office-31':
        base_path = 'datasets/Office31_mat/'
        source_loader, n_class, dim1, source_dataset = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader, target_dataset = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Digits':
        base_path = 'datasets/Digits/'
        source_loader, n_class, dim1, source_dataset = get_loader_source(args.batchsize, base_path, source_domain)
        target_loader, target_dataset = get_loader_target(args.batchsize, base_path, target_domain)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']
    else:
        raise Exception('No this dataset!')

    train(args, source_loader, 
          target_loader['train'], target_loader['query'], 
          n_class, dim1,
          source_dataset, target_dataset['train'],
    )
