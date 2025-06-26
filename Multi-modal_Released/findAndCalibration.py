import torch
import numpy as np
import torch.nn.functional as F

def findAndCalibration(threshold, evidences_all, nearest_indices, similarity_matrix, 
                      train_index, train_Y, args, classes_num, device, epoch, accuracies):
    """
    Identify noisy labels and perform calibration through similarity calculation.
    """
    sample_num = train_Y.shape[0]
    nearest_indices = torch.Tensor(nearest_indices).long().to(device)
    similarity_matrix = torch.Tensor(similarity_matrix).to(device)
    train_index = torch.Tensor(train_index).to(device).long()
    train_Y = train_Y.float()
    
    likes_all = torch.zeros(args.views, sample_num, device=device)
    changed_samples_ind = []
    
    statistics = []
    for v in range(args.views):
        # Calculate similarity between evidence distribution and one-hot labels
        evidences_v = F.softmax(evidences_all[v], dim=1)

        a = (train_Y).unsqueeze(1) + 1e-8
        b = evidences_v[nearest_indices[v]] + 1e-8
        b = torch.cat([evidences_v.unsqueeze(1), b], dim=1)

        # Calculate Jensen-Shannon divergence
        m = 0.5 * (a + b)
        js_div = 0.5 * torch.sum(a * (a.log() - m.log()), dim=-1) + \
                  0.5 * torch.sum(b * (b.log() - m.log()), dim=-1)
        
        # Replace NaN values with mean
        mean_value = torch.nanmean(js_div)
        js_div = torch.where(torch.isnan(js_div), mean_value, js_div)
    
        # Get neighbor similarities
        neighbor_similarity = torch.Tensor(similarity_matrix[v]).to(device)
        # Add self-similarity as first element
        neighbor_similarity = torch.cat([torch.ones(sample_num, 1).to(device), neighbor_similarity], dim=1)
        # Normalize similarities
        neighbor_similarity = F.softmax(neighbor_similarity, dim=1)

        likes_v = torch.sum(neighbor_similarity * js_div, dim=1)
        likes_all[v] = likes_v

        # Calculate statistics
        s_max = likes_all[v].max().item()
        s_min = likes_all[v].min().item()
        s_var = torch.var(likes_all[v]).item()
        statistics.append({"max": s_max, "min": s_min, "var": s_var})

    # Calculate fusion weights
    ranges = np.array([s["var"] for s in statistics])
    if np.isnan(ranges).any():
        print("NaN detected in variance calculation, using accuracy-based weights")
        weights = accuracies
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(args.views) / args.views
    else:
        variances = ranges / np.max(ranges)
        weights = variances / np.sum(variances)
    
    print("Fusion weights:", weights)

    # Combine evidence from all views
    likes_all_v = likes_all
    likes_all = torch.sum(likes_all * torch.Tensor(weights).to(device).unsqueeze(1), dim=0)
    likes_all = (likes_all - likes_all.min()) / (likes_all.max() - likes_all.min())

    # Print statistics
    view_mean = likes_all.mean().item()
    view_min = likes_all.min().item()
    view_max = likes_all.max().item()
    print(f"Combined likes_all - Mean: {view_mean:.4f}, Min: {view_min:.4f}, "
          f"Max: {view_max:.4f}, Range: {view_max-view_min:.4f}")

    # Find noisy samples using threshold
    error_index = (likes_all > threshold).nonzero(as_tuple=True)[0]
    sorted_indices = torch.argsort(likes_all)[:len(error_index)]
    true_index = sorted_indices

    # Generate new labels for noisy samples
    changed_label = torch.zeros(len(error_index), args.views, classes_num, device=device)
    neighbor_evidences = torch.zeros(len(error_index), args.views, args.k+1, classes_num, device=device)
    
    for v in range(args.views):
        # Get evidence for error samples and their neighbors
        neighbor_evidences[:, v, 0] = evidences_all[v][error_index]
        neighbor_indices = nearest_indices[v, error_index]
        neighbor_evidences[:, v, 1:] = evidences_all[v][neighbor_indices]
        
        # Calculate similarity-weighted new labels
        neighbor_similarity = similarity_matrix[v][error_index]
        neighbor_similarity = torch.cat([torch.ones(len(error_index), 1).to(device), neighbor_similarity], dim=1)
        neighbor_similarity = F.softmax(neighbor_similarity, dim=1).unsqueeze(-1)

        changed_label[:, v] = torch.sum(neighbor_similarity * neighbor_evidences[:, v], dim=1)

    # Weight according to view importance
    changed_label = F.softmax(changed_label, dim=2)
    changed_label = torch.sum(changed_label * torch.Tensor(weights).to(device).unsqueeze(1), dim=1)
    changed_label = torch.argmax(changed_label, dim=1)

    # Find samples with actually changed labels
    y_argmax = train_Y[error_index].argmax(dim=1)
    diff_after = torch.ne(changed_label, y_argmax)
    changed_samples_ind = error_index[diff_after]

    # Prepare return values
    sorted_indices = torch.argsort(likes_all)[:len(likes_all) - len(error_index)]
    true_index = train_index[sorted_indices]
    error_index = error_index[diff_after]
    
    # Randomly sample clean samples for mixing
    true_index = true_index[torch.randperm(len(true_index))][:len(error_index)]
    changed_label = changed_label[diff_after]

    return changed_samples_ind, true_index, error_index, changed_label