from torch import nn
from encoder import BertClf, ImageClf 
from loss import *
import torch
import torch.nn.functional as F

class TMNRR(nn.Module):
    """Trust-aware Multi-view Noise Resilient Representation learning model."""
    
    def __init__(self, args, classes, views, device, S=None, nearest_indices=None):
        super(TMNRR, self).__init__()
        
        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)
        self.classes = 101
        self.S = S
        self.nearest_indices = nearest_indices
        self.device = device

        # Initialize trust matrices for each view
        diagonal_matrix = torch.eye(classes)
        expanded_diagonal = diagonal_matrix.expand(views, classes, classes)
        self.matrixes = nn.Parameter(expanded_diagonal)

    def DS_Combin(self, alpha):
        """
        Dempster-Shafer combination of evidence from multiple views.
        
        Args:
            alpha: Dictionary of Dirichlet distribution parameters for each view
            
        Returns:
            Combined Dirichlet distribution parameters
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, args, txt, mask, segment, img, indexes, label, corrected_Y, epoch, error_index=None):
        # Extract features and predictions from both modalities
        txt_out, txt_f = self.txtclf(txt, mask, segment)
        img_out, img_f = self.imgclf(img)

        # Convert outputs to evidence
        evidence = {}
        evidence[0] = F.softplus(img_out)
        evidence[1] = F.softplus(txt_out)        
        transferred_evidence = self.transferByMatrix(evidence)

        alpha, t_alpha = {}, {}
        hasErrorSample = False
        l2_loss_clean = 0
        loss = 0
    
        argY = label.long()
        
        # Calculate losses for each view
        for v_num in range(2):
            alpha[v_num] = evidence[v_num] + 1
            t_alpha[v_num] = transferred_evidence[v_num] + 1
            loss += ce_loss(label, t_alpha[v_num], 101, epoch, args.epochs, self.device)

            # Add corrected loss for noisy samples during training
            if self.training:
                if error_index is not None and error_index.shape[0] > 0:
                    l2_v, hasErrorSample = corrected_loss(corrected_Y, alpha[v_num], indexes, error_index, self.device)
                    if hasErrorSample:
                        l2_loss_clean += l2_v

            U = 101 / torch.sum(t_alpha[v_num], dim=1)
            conf = (1 - U).to(self.device)
            loss += args.lamb * conf_loss(
                conf, self.matrixes[v_num], indexes, argY, self.classes, self.device
            )
        
        # Combine evidence from all views
        t_alpha_a = self.DS_Combin(t_alpha) 
        loss += ce_loss(label, t_alpha_a, 101, epoch, args.epochs, self.device)
        
        if epoch > args.start_correct:  # warm-start 
            loss += args.gamma * consistent_view_loss(self.matrixes, indexes, self.device)
            
        loss = torch.mean(loss)
        # print("loss: ", loss)
        if error_index is not None and error_index.shape[0] > 0 and hasErrorSample:
            loss += 10 * torch.mean(l2_loss_clean)

        evidence_a = t_alpha_a - 1
        return evidence, transferred_evidence, t_alpha, t_alpha_a, loss, evidence_a

    def transferByMatrix(self, evidence):
        """Transfer evidence through trust matrices."""
        transferred_evidence = {}
        if self.training:
            for v in range(2):
                selected_matrixes = self.matrixes[v]
                result = torch.matmul(evidence[v].unsqueeze(1), selected_matrixes).squeeze(1)
                transferred_evidence[v] = result
        else:
            for v in range(2):
                transferred_evidence[v] = evidence[v]
        return transferred_evidence