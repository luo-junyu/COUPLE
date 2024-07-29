import torch
import torch.nn.functional as F
import faiss
from sklearn import metrics
import warnings
import numpy as np
from munkres import Munkres
import random
import os

def seed_setting(seed=2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_centers(x, psedo_labels, num_cluster):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        weight = psedo_labels.T
    else:
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1
    weight = weight.float()
    weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
    centers = torch.mm(weight, x)
    centers = F.normalize(centers, dim=1)
    return centers

@torch.no_grad()
def psedo_labeling(num_cluster, batch_features, centers):
    l2_normalize =True
    torch.cuda.empty_cache()
    if l2_normalize:
        batch_features = F.normalize(batch_features, dim=1)
        batch_features_cpu = batch_features.cpu()
    centers_cpu = centers.cpu()
    btarget_cen_similarity = F.cosine_similarity(batch_features_cpu.unsqueeze(1), centers_cpu.unsqueeze(0), dim=2)
    relation = torch.zeros(num_cluster, dtype=torch.int64) - 1
    sorted_indices = torch.argsort(btarget_cen_similarity, dim=1, descending=True)
    new_cluster_labels = sorted_indices[:, 0]

    return new_cluster_labels


def clustering(features: torch.Tensor, n_clusters: int):
    x_np = features.numpy().astype('float32')
    dim = features.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=n_clusters, seed=2023, gpu=1, niter=100, verbose=False, nredo=5,
                          min_points_per_centroid=1, spherical=True)
    kmeans.train(x_np, init_centroids=None)
    centroids_np = kmeans.centroids
    centroids = torch.from_numpy(centroids_np)
    _, plabels_np = kmeans.index.search(x_np, 1)
    plabels = torch.from_numpy(plabels_np)

    return plabels, centroids

def evaluate_clustering(label, pred, eval_metric=['nmi', 'acc', 'ari'], phase='train'):
    mask = (label != -1)
    label = label[mask]
    pred = pred[mask]
    results = {}
    if 'nmi' in eval_metric:
        nmi = metrics.normalized_mutual_info_score(label, pred, average_method='arithmetic')
        results[f'{phase}_nmi'] = nmi
    if 'ari' in eval_metric:
        ari = metrics.adjusted_rand_score(label, pred)
        results[f'{phase}_ari'] = ari
    if 'f' in eval_metric:
        f = metrics.fowlkes_mallows_score(label, pred)
        results[f'{phase}_f'] = f
    if 'acc' in eval_metric:
        n_clusters = len(set(label))
        if n_clusters == len(set(pred)):
            pred_adjusted = get_y_preds(label, pred, n_clusters=n_clusters)
            acc = metrics.accuracy_score(pred_adjusted, label)
        else:
            acc = 0.
            warnings.warn('TODO: the number of classes is not equal...')
        results[f'{phase}_acc'] = acc
    return results

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def compute_cluster_loss(q_centers,
                         k_centers,
                         temperature,
                         psedo_labels,
                         num_cluster):
    d_q = q_centers.mm(q_centers.T) / temperature
    d_k = (q_centers * k_centers).sum(dim=1) / temperature
    d_q = d_q.float()
    d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k
    zero_classes = torch.arange(num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                             num_cluster), dim=0) == 0]
    mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)
    pos = d_q.diag(0)
    mask = torch.ones((num_cluster, num_cluster))
    mask = mask.fill_diagonal_(0).bool()
    neg = d_q[mask].reshape(-1, num_cluster - 1)
    loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    loss = loss.sum() / (num_cluster - len(zero_classes))
    return loss