import numpy as np
from scipy.ndimage import maximum_filter, label
import torch
from utils3d.gaussian_smoothing import GaussianSmoothing
from torch.nn import functional as F
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib.colors import rgb_to_hsv
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding

def fn_get_topk(attention_map, K=1, largest=True):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
        topk_value_list.append(topk_value[i])
    return topk_coord_list, topk_value_list


def fn_get_topk_plus(attention_map, K=1, threshold=0.8):
    H, W = attention_map.size()
    attention_map_detach = attention_map.view(H * W)  # .detach()
    topk_value, topk_index = attention_map_detach.topk(H * W, dim=0, largest=True, sorted=True)
    topk_coord_list = []
    topk_value_list = []
    threshold_coord_list = []
    threshold_value_list = []
    for i in range(len(topk_index)):
        index = topk_index[i].cpu().numpy()
        coord = index // W, index % W
        if i < K:  # topk_value[i]> threshold and
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[i])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        elif i < 4 * K or topk_value[i] > threshold:
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[i])
        if i > 4 * K and topk_value[i] < threshold: break
    """if len(topk_value_list)==0:
            index = topk_index[0].cpu().numpy()
            coord = index // W, index % W
            topk_coord_list.append(coord)
            topk_value_list.append(topk_value[0])
            threshold_coord_list.append(coord)
            threshold_value_list.append(topk_value[0])"""

    return topk_coord_list, topk_value_list, threshold_coord_list, threshold_value_list


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map


def fn_show_attention_plus_3(
        cross_attention_maps,
        self_attention_maps,
        indices=None,
        attention_res=64,
        smooth_attentions=True,
        threshold=1.0
):  
    # cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    # cross_attention_maps = cross_attention_maps * 100
    # cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
    # # Shift indices since we removed the first token
    # indices = [index - 1 for index in indices]
    # cross_attention_maps = cross_attention_maps[:, :, indices].sum(dim=-1)
    # print(cross_attention_maps.shape)

    if smooth_attentions: cross_attention_maps = fn_smoothing_func(cross_attention_maps)
    cross_mask = fn_get_otsu_mask(cross_attention_maps, threshold)

    agg_self_attn_map = torch.zeros_like(cross_attention_maps)
    # print(cross_attn_map.shape, agg_self_attn_map.shape)
    for j in range(attention_res):
        for k in range(attention_res):
            if cross_mask[j, k]:
                cross_attn_value = cross_attention_maps[j, k]
                self_attn_map = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                agg_self_attn_map = agg_self_attn_map + cross_attn_value * self_attn_map

    agg_self_attn_map = (agg_self_attn_map - agg_self_attn_map.min()) / \
                                            ( agg_self_attn_map.max() - agg_self_attn_map.min() + 1e-6)
    # print(agg_self_attn_map.sum(),agg_self_attn_map.max(),agg_self_attn_map.min())
    self_mask = fn_get_otsu_mask(agg_self_attn_map, threshold)  # fine-tune the mask size

    return cross_attention_maps, agg_self_attn_map, self_mask, cross_mask


def fn_show_attention_plus_seg4diff(
        cross_attention_maps,
        self_attention_maps,
        attention_res=64,
        smooth_attentions=True,
        indices=None,
        threshold=1.0
):
    ##############  preprocessing, will make mask smaller  ###############
    cross_attention_maps = cross_attention_maps[:, :, :, 1:-1]
    # cross_attention_maps = cross_attention_maps * 100
    # cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
    # Shift indices since we removed the first token
    # indices = [index - 1 for index in indices]
    indices = indices - 1

    max_values, max_indices=torch.max(cross_attention_maps, dim=-1)
    cross_mask = torch.zeros_like(max_values)
    cross_mask[max_indices == indices] = 1
    cross_mask = torch.sum(cross_mask, dim=0)  # if any layer's positive then it's positive
    cross_mask = cross_mask.to(torch.bool)
    ########################  cross_attn_map  ############################

    # if smooth_attentions: cross_attn_map = fn_smoothing_func(cross_attn_map)
    # cross_mask = fn_get_otsu_mask(cross_attn_map, threshold)
    cross_attn_map = torch.sum(cross_attention_maps[:,:, :, indices],dim=0)

    agg_self_attn_map = torch.zeros_like(cross_attn_map)
    # print(cross_attn_map.shape, agg_self_attn_map.shape)
    for j in range(attention_res):
        for k in range(attention_res):
            if cross_mask[j, k]:
                cross_attn_value = cross_attn_map[j, k]
                self_attn_map = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                agg_self_attn_map = agg_self_attn_map + cross_attn_value * self_attn_map

    agg_self_attn_map = (agg_self_attn_map - agg_self_attn_map.min()) / \
                                            ( agg_self_attn_map.max() - agg_self_attn_map.min() + 1e-6)
    # print(agg_self_attn_map.sum(),agg_self_attn_map.max(),agg_self_attn_map.min())
    self_mask = fn_get_otsu_mask(agg_self_attn_map, threshold)  # fine-tune the mask size

    return cross_attn_map, agg_self_attn_map, self_mask


def fn_show_attention(
        cross_attention_maps,
        self_attention_maps,
        # cross_attention_map_cache,
        indices=None,
        attention_res=64,
        smooth_attentions=True,
        threshold=1.0
):
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]
    cross_attention_maps = cross_attention_maps[:, :, indices].sum(dim=-1)
    # print(cross_attention_maps.shape)

    if smooth_attentions: cross_attention_maps = fn_smoothing_func(cross_attention_maps)
    # if cross_attention_map_cache is not None:cross_attention_maps = cross_attention_map_cache + cross_attention_maps
    cross_mask = fn_get_otsu_mask(cross_attention_maps, threshold)

    agg_self_attn_map = torch.zeros_like(cross_attention_maps)
    # print(cross_attn_map.shape, agg_self_attn_map.shape)
    for j in range(attention_res):
        for k in range(attention_res):
            if cross_mask[j, k]:
                cross_attn_value = cross_attention_maps[j, k]
                self_attn_map = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                agg_self_attn_map = agg_self_attn_map + cross_attn_value * self_attn_map

    agg_self_attn_map = (agg_self_attn_map - agg_self_attn_map.min()) / \
                                            ( agg_self_attn_map.max() - agg_self_attn_map.min() + 1e-6)
    # print(agg_self_attn_map.sum(),agg_self_attn_map.max(),agg_self_attn_map.min())
    
    self_mask = fn_get_otsu_mask(agg_self_attn_map, threshold)  # fine-tune the mask size

    return cross_attention_maps, agg_self_attn_map, self_mask


def fn_get_otsu_mask(x,scalar=1.0):
    x_numpy = x.to(torch.float16)
    x_numpy = x_numpy.cpu().detach().numpy()
    x_numpy = x_numpy * 255
    x_numpy = x_numpy.astype(np.uint16)

    opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opencv_threshold = opencv_threshold * 1. / 255.

    otsu_mask = torch.where(
        x < opencv_threshold* scalar,
        torch.tensor(0, dtype=x.dtype, device=x.device),
        torch.tensor(1, dtype=x.dtype, device=x.device))

    return otsu_mask.to(torch.int).cpu().numpy()


def fn_clean_mask(otsu_mask, x, y):
    otsu_mask=torch.from_numpy(otsu_mask)
    H, W = otsu_mask.shape
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)

    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask


def db_scan(self_attn_feats_mean, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', min_cluster_size=10):
    # self_attn_feats_mean = (self_attn_feats_mean + self_attn_feats_mean.T) / 2
    # print(self_attn_feats_mean.shape)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(self_attn_feats_mean)
    cluster_labels = clustering.labels_
    # print(cluster_labels.max(), cluster_labels.min())
    n_clusters = len(set(cluster_labels)) - (1 if -1 in clustering.labels_ else 0)

    small_clusters = [i for i in range(n_clusters) if (clustering.labels_ == i).sum() < min_cluster_size]

    for i in small_clusters:
        cluster_labels[clustering.labels_ == i] = -1
        n_clusters -= 1

    # Re number clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    for i, cluster in enumerate(set(cluster_labels)):
        if cluster == -1:
            continue
        cluster_labels[cluster_labels == cluster] = i

    return cluster_labels, n_clusters 


def db_scan_feat(self_attn_feats_mean, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', min_cluster_size=10):
    # for PCA features, both label -1 and label 0 correspond to background
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(self_attn_feats_mean)
    cluster_labels = clustering.labels_
    # print(cluster_labels.max(), cluster_labels.min())
    n_clusters = len(set(cluster_labels)) - (1 if -1 in clustering.labels_ else 0)

    # reduce small clusters
    small_clusters = [i for i in range(n_clusters) if (clustering.labels_ == i).sum() < min_cluster_size]
    for i in small_clusters:
        cluster_labels[clustering.labels_ == i] = -1
        n_clusters -= 1

    # Re number clusters
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    # merge 0 and -1
    for i, cluster in enumerate(set(cluster_labels)):
        # print(i, cluster)
        if cluster == -1:
            continue
        else:
            cluster_labels[cluster_labels == cluster] = i-1

    return cluster_labels, n_clusters-1 


def remove_sparse_blobs(grid, min_blob_size=10):
    grid = grid.reshape(32, 32)
    unique_masks = np.unique(grid)
    unique_masks = unique_masks[unique_masks >= 0]
    # Create a copy of the grid to modify
    updated_grid = grid.copy()
    # Process each unique mask
    for mask_id in unique_masks:
        # Create a binary mask for the current id
        binary_mask = (grid == mask_id)
        # Label connected components in the binary mask
        labeled_array, num_features = label(binary_mask)
        # Boolean to track if all blobs are less than 10 pixels
        all_blobs_small = True
        # Check each labeled feature
        for i in range(1, num_features + 1):
            blob_size = np.sum(labeled_array == i)
            if blob_size >= min_blob_size:
                all_blobs_small = False
                break
        # If all blobs in this mask are smaller than 10 pixels, mark those regions as -1
        if all_blobs_small:
            updated_grid[binary_mask] = -1

    # Re number clusters
    unique_masks = np.unique(updated_grid)
    unique_masks = unique_masks[unique_masks >= 0]
    n_cluster_new = len(unique_masks)

    
    for i, cluster in enumerate(unique_masks):
        updated_grid[updated_grid == cluster] = i

    return torch.tensor(updated_grid).flatten(), n_cluster_new


def compute_pca(features, n_components=3, normalize=False, spectral=True):
    # min max normalization needed
    if normalize:
        features = (features - features.min()) / (features.max() - features.min())
    H, W, C = features.shape  # C
    features = features.reshape(W*H, -1).cpu().numpy()

    # tsne = TSNE(n_components=n_components, perplexity=30, random_state=42, init='random', learning_rate='auto')
    # X_tsne = tsne.fit_transform(features)  # shape (N, 3)
    # X_tsne = X_tsne.reshape(H, W, n_components)
    # features = (X_tsne - X_tsne.min()) / (X_tsne.max() - X_tsne.min())

    # reducer = umap.UMAP(n_components=n_components, n_neighbors=20, min_dist=0.1, random_state=42,n_jobs=1)
    # X_umap = reducer.fit_transform(features)
    # X_umap = X_umap.reshape(H, W, n_components)
    # features = (X_umap - X_umap.min()) / (X_umap.max() - X_umap.min())

    # isomap = Isomap(n_components=n_components, n_neighbors=10)
    # X_iso = isomap.fit_transform(features)
    # X_iso = X_iso.reshape(H, W, n_components)
    # features = (X_iso - X_iso.min()) / (X_iso.max() - X_iso.min())


    if spectral:
        se = SpectralEmbedding(n_components=n_components, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
        X_se = se.fit_transform(features)
        X_se = X_se.reshape(H, W, n_components)
        features = (X_se - X_se.min()) / (X_se.max() - X_se.min())
    else:
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        # features = features.reshape(H, W, -1)

    return features



def fn_show_attention_plus_2(
        cross_attention_maps,
        self_attention_maps,
        indices,
        K=1,
        attention_res=16,
        smooth_attentions=True,
):
    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]
    cross_attention_map_list = []
    cross_attention_map_list_for_show = []
    self_attention_map_top_list = []
    otsu_masks = []
    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        topk_coord_list, topk_value_list = fn_get_topk(cross_attention_map_per_token, K=K)
        self_attn_map_top1 = self_attention_maps[topk_coord_list[0][0], topk_coord_list[0][1]].view(attention_res, attention_res).contiguous()
        # print(self_attn_map_top1.shape, self_attention_maps.shape)
        cross_attention_map_list_for_show.append(cross_attention_map_per_token.to(torch.float16).cpu().detach().numpy())
        self_attention_map_top_list.append(self_attn_map_top1.to(torch.float16).cpu().detach().numpy())
        cross_attention_map_list.append(cross_attention_map_per_token)

        # -----------------------------------
        # clean cross_attention_map_cur_token
        # -----------------------------------
        clean_cross_attention_map_per_token_mask = fn_get_otsu_mask(cross_attention_map_per_token)
        # clean_cross_attention_map_per_token_mask = fn_clean_mask(clean_cross_attention_map_per_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
        otsu_masks.append(clean_cross_attention_map_per_token_mask)

    self_attention_map_list = []
    self_attention_map_list_list=[]
    for i in range(len(cross_attention_map_list)):
        cross_attn_map_cur_token = cross_attention_map_list[i]
        self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
        mask_cur_token = otsu_masks[i]
        cross_attn_value_cur_token_sum = 0
        self_attention_map_per_token_list = []
        self_atten_map_list=[]
        for j in range(attention_res):
            for k in range(attention_res):
                if mask_cur_token[j, k] == 0: continue
                cross_attn_value_cur_token = cross_attn_map_cur_token[j, k]
                cross_attn_value_cur_token_sum = cross_attn_value_cur_token_sum + cross_attn_value_cur_token
                self_attn_map_cur_position = self_attention_maps[j, k].view(attention_res, attention_res).contiguous()
                self_attn_map_cur_token = self_attn_map_cur_token + cross_attn_value_cur_token * self_attn_map_cur_position
                self_atten_map_list.append(self_attn_map_cur_position)
        self_attention_map_per_token_list.append(self_attn_map_cur_token)
        self_attention_map_list_list.append(self_atten_map_list)
        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / cross_attn_value_cur_token_sum
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
                                            (
                                                    self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        # print(norm_self_attention_map_per_token.shape)
        self_attention_map_list.append(norm_self_attention_map_per_token.to(torch.float16).cpu().detach().numpy())

    # tensor to numpy
    # cross_attention_map_numpy = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    # self_attention_map_numpy = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    return cross_attention_map_list_for_show, self_attention_map_list, self_attention_map_top_list