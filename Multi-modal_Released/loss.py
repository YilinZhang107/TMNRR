import torch
import torch.nn.functional as F

def KL(alpha, c, device):
    """Calculate KL divergence for Dirichlet distribution."""
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def corrected_loss(corrected_Y, alpha, indexes, error_index, device):
    """Calculate L2 loss for corrected labels on noisy samples."""
    label = F.one_hot(corrected_Y, num_classes=101).float()
    label = torch.Tensor(label).to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Identify noisy samples in current batch
    error_mask = torch.isin(indexes, error_index).to(device)
    
    # Calculate L2 loss for noisy samples
    noise_loss = None
    if error_mask.any():
        noise_loss = torch.sum(
            (label[error_mask] - alpha[error_mask] / S[error_mask]) ** 2, 
            dim=1, keepdim=True
        )
    return noise_loss, error_mask.sum() > 0

def ce_loss(label, alpha, c, global_step, annealing_step, device):
    """Adjusted cross-entropy loss based on Dirichlet distribution."""
    # Convert label to one-hot
    label = F.one_hot(label, num_classes=c).float()

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    annealing_coef = min(1, global_step / annealing_step)
    B = annealing_coef * KL(alp, c, device)
    A = A + B
    return A

def similarity_loss(nearest_similarities, near_indices, matrix, indexes, device):
    """Similarity-based constrained noise transfer matrix loss."""
    k = near_indices.shape[-1]
    classes = matrix.shape[-1]
    bs = len(indexes)
    x_indices = indexes.long()
    x_indices = x_indices.cpu().numpy()
    y_indices = torch.Tensor(near_indices[x_indices]).to(device).long()

    # Get similarity values
    S_values = torch.Tensor(nearest_similarities[x_indices]).to(device)

    m1 = matrix[x_indices]
    m1 = m1.unsqueeze(1).expand(bs, k, classes, classes)
    m2 = matrix[y_indices]

    # Only count non-diagonal elements
    m1 = m1 * (1 - torch.eye(classes, device=device).unsqueeze(0).unsqueeze(0))
    m2 = m2 * (1 - torch.eye(classes, device=device).unsqueeze(0).unsqueeze(0))

    # Calculate similarity loss for each pair
    distance = torch.norm(m1 - m2 + 1e-10, dim=(2, 3))
    pairwise_loss = S_values * distance

    loss = torch.sum(pairwise_loss, dim=1, keepdim=True) / k
    return loss

def conf_loss(conf_a, T, indexes, y, class_num, device):
    """Confidence-based loss ensuring confidence is proportional to trust matrix diagonal."""
    # Calculate mean confidence for all classes
    classes_conf = torch.zeros(class_num, device=device)
    class_counts = torch.bincount(y, minlength=class_num).float()
    classes_conf = torch.scatter_add(classes_conf, dim=0, index=y, src=conf_a)
    classes_conf = classes_conf / (class_counts + 1e-5)

    # Get diagonal elements from shared matrix T
    diagonal_elements = torch.diag(T)
    diagonal_elements = diagonal_elements.unsqueeze(0).expand(len(conf_a), -1)

    # Create one-hot encoding for true categories
    one_hot_y = torch.zeros(len(conf_a), class_num, device=device)
    one_hot_y.scatter_(1, y.view(-1, 1), 1)
    
    loss = ((conf_a.view(-1, 1) - diagonal_elements) ** 2 * one_hot_y).sum(dim=1) + \
           ((classes_conf.view(1, -1) - diagonal_elements) ** 2 * (1 - one_hot_y)).sum(dim=1)

    return loss.reshape(-1, 1)

def consistent_view_loss(T, indexes, device):
    """Loss function for inter-view consistency of trust matrices."""
    v = T.shape[0]
    K = T.shape[2]

    reshaped_T = T.view(v, -1)
    # Calculate squared differences between all pairs of matrices
    squared_diff = (reshaped_T.unsqueeze(1) - reshaped_T.unsqueeze(0)).pow(2)
    # Apply mask to exclude self-comparisons
    mask = torch.eye(v, device=device).bool()
    squared_diff_masked = squared_diff[~mask]
    loss = squared_diff_masked.sum() / (2 * v * (v-1) * K * K)

    return loss.view(1)










