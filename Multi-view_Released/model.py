from loss import *
import torch.nn as nn

class TMNRR(nn.Module):
    def __init__(self, classes, views, classifier_dims, sample_num, device, lambda_epochs=1, S=None, nearest_indices=None):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMNRR, self).__init__()
        self.classes = classes
        self.views = views
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        # Create a diagonal matrix with diagonal elements initially 1
        diagonal_matrix = torch.eye(classes)
        # Expanding the diagonal matrix into dimensions (n, v, k, k)
        expanded_diagonal = diagonal_matrix.expand(views, sample_num, classes, classes)
        self.matrixes = nn.Parameter(expanded_diagonal)
        self.S = S
        self.near_indices = nearest_indices
        self.device = device

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
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

    def forward(self, X, y, corrected_Y, indexes, global_step, lamb=1,  gamma=1, error_index=None, l2=25):
        indexes = indexes.long().to(self.device)
        evidence = self.infer(X)  
        # Calculate how many samples changed labels before and after the transfer
        predicted = {}
        for v in range(len(X)):
            predicted[v] = torch.argmax(evidence[v], dim=1)  #

        transferred_evidence = self.transferByMatrix(evidence, indexes)
        different_num = 0
        for v in range(len(X)):
            result = torch.argmax(transferred_evidence[v], dim=1)  #
            different_num += torch.sum(result != predicted[v]).item()
        transferred_alpha, alpha = {}, {}
        loss = 0
        l2_loss_clean = 0
        hasErrorSample = False
        argY = torch.argmax(y, dim=1).long()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            transferred_alpha[v_num] = transferred_evidence[v_num] + 1  
            if self.training:
                loss += ce_loss(y, transferred_alpha[v_num], self.classes, global_step, self.lambda_epochs, self.device, indexes, error_index, l2)
                if error_index is not None and error_index.shape[0] > 0:
                    l2_v, hasErrorSample = corrected_loss(corrected_Y, alpha[v_num], indexes, error_index, self.device)
                    if hasErrorSample:
                        l2_loss_clean += l2_v

                U = self.classes / torch.sum(transferred_alpha[v_num], dim=1)
                conf = (1 - U).to(self.device)
                loss += lamb * conf_loss(conf, self.matrixes[v_num], indexes, argY, self.classes, self.device)
                loss += lamb * similarity_loss(self.S[v_num], self.near_indices[v_num], self.matrixes[v_num], indexes, self.device)
        evidence_a = None
        
        alpha_a = self.DS_Combin(transferred_alpha)
        evidence_a = alpha_a - 1 
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs, self.device, indexes, error_index, l2)

        if global_step > 9:  # warm-start
            loss += gamma * consistent_view_loss(self.matrixes, indexes, self.device)
        
        loss = torch.mean(loss) 
        if error_index is not None and error_index.shape[0] > 0 and hasErrorSample:
            loss += l2 * torch.mean(l2_loss_clean)
      
        del alpha, alpha_a
        torch.cuda.empty_cache()

        return transferred_evidence, evidence_a, loss, evidence, different_num/len(X)

    def transferByMatrix(self, evidence, indexes):
        """
        The transfer matrix is different for each sample, and the evidence
        obtained from each perspective is passed through the noise transfer matrix
        :return: new evidence
        """
        transferred_evidence = {}
        if self.training:
            for v in range(self.views):
                selected_matrixes = self.matrixes[v][indexes]
                result = torch.matmul(evidence[v].unsqueeze(1), selected_matrixes).squeeze(1)
                transferred_evidence[v] = result
        else:
            for v in range(self.views):
                transferred_evidence[v] = evidence[v]
        return transferred_evidence

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h

