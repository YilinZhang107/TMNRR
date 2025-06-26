import torch
import numpy as np
import torch.nn.functional as F


"""
Identify noisy labels and perform calibration through similarity calculation.
The adjustable parameters here are:
1. args.k: Select the nearest k samples.
2. threshold: Select the similarity threshold.
3. warm_up: From which epoch to start calibration.
"""
def findAndCalibration(threshold, evidences_all, nearest_indices, similarity_matrix, train_index, train_Y, Y_true, args, classes_num, device, epoch):

    train_index = torch.Tensor(train_index).to(device).long()
    Y = torch.Tensor(train_Y).to(device).float()
    likes_all = torch.zeros(args.views, train_index.shape[0], device=device) 
    changed_samples_ind = []

    statistics = []
    for v in range(args.views):
        # Calculate the similarity between the evidence distribution of all samples and the one-hot labels.
        evidences_v = F.softmax(evidences_all[v], dim=1)

        S = (evidences_all[v][train_index] + 1).sum(dim=1)
        U = (classes_num / S).mean()

        a = (Y[train_index]).unsqueeze(1) + 1e-8
        b = evidences_v[nearest_indices[v, train_index]] + 1e-8

        b = torch.cat([evidences_v[train_index].unsqueeze(1), b], dim=1)

        m = 0.5 * (a + b)
        js_div = 0.5 * torch.sum(a * (a.log() - m.log()), dim=-1) + 0.5 * torch.sum(b * (b.log() - m.log()), dim=-1)

        # Use the similarity between neighbors for weighting, and extract the feature similarity between each sample and its neighbors from the similarity_matrix.
        train_rows = train_index.unsqueeze(1).expand(-1, args.k)  # [n, k]

        neighbor_similiarity = similarity_matrix[v][train_rows, nearest_indices[v][train_index]]
        neighbor_similiarity = torch.cat([torch.ones_like(neighbor_similiarity[:, :1]), neighbor_similiarity], dim=1)
        neighbor_similiarity = F.softmax(neighbor_similiarity, dim=1)

        likes_v = js_div
        likes_all[v] = torch.sum(neighbor_similiarity * likes_v, dim=1)

        s_max = max(likes_v.mean(dim=1)).item()
        s_min = min(likes_v.mean(dim=1)).item()

        s_var = torch.var(likes_v.mean(dim=1)).item()
        statistics.append({"max": s_max, "min": s_min, "var": s_var}) 

    ranges = np.array([s["var"] for s in statistics])  # Use of variance as a discriminator
    variances = ranges / np.max(ranges)
    weights = variances / np.sum(variances)

    likes_all = torch.sum(likes_all * torch.Tensor(weights).to(device).unsqueeze(1), dim=0)

    likes_all = (likes_all - likes_all.min()) / (likes_all.max() - likes_all.min())

    error_index = (likes_all > threshold).nonzero(as_tuple=True)[0]  # Use the method of Jensen - Shannon divergence.
    error_index = train_index[error_index]  # Extract the corresponding indices from `train_index` according to `error_index`.
    # Get the indices of the smallest n values.
    sorted_indices = torch.argsort(likes_all)[:len(error_index)]
    true_index = train_index[sorted_indices]

    changed_label = torch.zeros(len(error_index), args.views, classes_num, device=device)
    neighbor_evidences = torch.zeros(len(error_index), args.views, args.k+1, classes_num, device=device)
    for v in range(args.views):
        # Calculate the mean of the evidence of itself and its k nearest samples.
        neighbor_evidences[:, v, 0] = evidences_all[v][error_index]  # Obtain the evidence of the current perspective for all error indices.

        neighbor_indices = nearest_indices[v, error_index]  # [len(error_index), k]

        neighbor_evidences[:, v, 1:] = evidences_all[v][neighbor_indices]
        # Calculate the new labels using similarity-weighted calculation.
        error_rows = error_index.unsqueeze(1).expand(-1, args.k)  # [n, k]
        neighbor_similiarity = similarity_matrix[v][error_rows, neighbor_indices]
        neighbor_similiarity = torch.cat([torch.ones_like(neighbor_similiarity[:, :1]), neighbor_similiarity], dim=1)
        neighbor_similiarity = F.softmax(neighbor_similiarity, dim=1).unsqueeze(-1)

        changed_label[:, v] = torch.sum(neighbor_similiarity * neighbor_evidences[:, v], dim=1)  # Weighted according to similarity with neighbors

    # Weight according to the variance.
    changed_label = F.softmax(changed_label, dim=2)
    changed_label = torch.sum(changed_label * torch.Tensor(weights).to(device).unsqueeze(1), dim=1)

    changed_label = torch.nn.functional.one_hot(torch.argmax(changed_label, dim=1), num_classes=classes_num)  

    changed = torch.argmax(changed_label, dim=1)
    y_argmax = Y[error_index].argmax(dim=1)
    Y_true = torch.Tensor(Y_true).to(device)
    yt = torch.argmax(Y_true[error_index], dim=1)

    diff_after = torch.ne(changed, y_argmax)
    changed_samples_ind = error_index[diff_after]

    sorted_indices = torch.argsort(likes_all)[:len(likes_all) - len(error_index)]  #  Get indexes other than error_index
    true_index = train_index[sorted_indices]
    error_index = error_index[diff_after]
    # Randomly sample len(error_index) samples from true_index
    true_index = true_index[torch.randperm(len(true_index))][:len(error_index)]
    changed_label = changed_label[diff_after]

    return changed_samples_ind, torch.Tensor(true_index).to(device), torch.Tensor(error_index).to(device), changed_label
