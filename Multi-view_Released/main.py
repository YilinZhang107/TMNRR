import os
import pickle
import torch
from torch.utils.data import DataLoader, Subset
from findAndCalibration import findAndCalibration
from utils.data import MultiViewDataset
from utils.config import loadConfig
import argparse
from model import TMNRR
import numpy as np
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
The official implementation of TMNR^2

Note: The default hyperparameters for each dataset are configured in utils.config, which may cause the parameters input via the command line to be ineffective.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='input batch size for training', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs to train', default=300)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)  # Leaves default learning rate is 0.003
parser.add_argument('--dataset', type=str, help='PIE, Caltech101, UCI, BBC, Leaves, CUB', default="Caltech101")
args = parser.parse_args()

noise_ratios = np.arange(0.0, 0.55, 0.10)  # to run with the noise rate of 0, 0.1, 0.2, 0.3, 0.4, 0.5
saveAcc = False

def main():
    global args

    def experiment(ratio, repeat_num, threshold):

        train_loader, test_loader, classes_num, X_np, train_Y_origin, train_index, sample_num, Y_true = import_and_load_data(ratio)

        """
        Similarity maps are computed for each view
        """
        similarity_matrix = torch.zeros(args.views, sample_num, sample_num).to(device)
        train_X = {}
        nearest_indices = torch.zeros((args.views, sample_num, args.k), dtype=torch.long, device=device)
        for v in range(args.views):
            train_X[v] = torch.Tensor(X_np[v]).to(device)
            distance_squared = (torch.cdist(train_X[v], train_X[v], p=2) ** 2)

            # 为每个样本找到 k 个最近邻（不包括自己）
            k_for_sigma = min(args.k, sample_num - 1)  # 用于计算 sigma 的 k 值
            sigma = torch.zeros(sample_num, 1).to(device)
            # 同时记录排序后的索引，避免后续重复排序
            nearest_indices_view = torch.zeros((sample_num, args.k), dtype=torch.long, device=device)
            
            for j in range(sample_num):
                # 获取距离按升序排列的索引（排除自身）
                sorted_distances, sorted_indices = torch.sort(distance_squared[j])
                # 取前 k 个最近邻距离（排除自身，所以从索引1开始）
                k_nearest_distances = sorted_distances[1:k_for_sigma+1]
                # 计算平均距离作为 sigma
                sigma[j] = torch.mean(k_nearest_distances)
                # 同时记录最相似的k个样本的索引(不包括自己)
                nearest_indices_view[j] = sorted_indices[1:args.k+1]
            # 防止 sigma 过小导致数值不稳定
            sigma = torch.clamp(sigma, min=1e-5)
            # 使用自适应 sigma 计算相似度
            # 广播 sigma 到每一行
            similarity_matrix[v] = torch.exp(-distance_squared / (sigma ** 2))  # 
            # 直接将前面已经计算好的最近邻索引赋值给nearest_indices
            nearest_indices[v] = nearest_indices_view

        """
        In each view, find the sample that is most similar to each sample
        """
        # k = args.k
        # nearest_indices = torch.zeros((args.views, sample_num, k), dtype=torch.long, device=device)  
        # for i in range(args.views):
        #     for j in range(sample_num):
        #         sorted_indices = torch.argsort(similarity_matrix[i][j], descending=True)  # 进行降序排序
        #         nearest_indices[i, j] = sorted_indices[1:k+1]

        """train"""
        history = np.zeros((args.epochs, 2))
        start_time = time.time()
        model = TMNRR(classes_num, args.views, args.dims, sample_num, device, args.lambda_epochs, similarity_matrix, nearest_indices)

        optimizer1 = torch.optim.Adam(model.Classifiers.parameters(), lr=args.lr, weight_decay=1e-5)

        optimizer2 = torch.optim.Adam([model.matrixes], lr=1e-3, weight_decay=1e-5)
        model.to(device)
        # print(model)

        total_error_index = None
        train_Y = train_Y_origin 
        for epoch in tqdm(range(0, args.epochs)):
            model.train()
            total_loss = 0
            YchangedNum_sum = 0

            evidences_all = torch.zeros(args.views, sample_num, classes_num).to(device)

            for X, Y, indexes, corrected_Y, _ in train_loader:
                Y = Y.to(device)
                indexes = indexes.long().to(device)
                for v_num in range(len(X)):
                    X[v_num] = X[v_num].to(device)
                evidences, evidence_a, loss, evidences_notrans, YchangedNum = model(X, Y, corrected_Y, indexes, epoch, args.lamb, args.gamma, total_error_index, args.l2)
                YchangedNum_sum += YchangedNum

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                if epoch > args.warmUp:  # note warm-start
                    optimizer2.step()

                # such that the updated matrix satisfies its own constraints
                t = model.matrixes.detach()
                t = torch.clamp(t, min=0)
                normalized_matrixes = t / t.sum(dim=-1, keepdim=True)
                with torch.no_grad():
                    model.matrixes.data = normalized_matrixes

                total_loss += loss * len(Y)

                for v_num in range(len(X)):
                    evidences_all[v_num][indexes] = evidences_notrans[v_num]

            acc = test(model, test_loader, epoch)
            # if epoch % 5 == 0:
            # print('Epoch {} ====> test_acc: {:.4f}, loss = {}'.format(epoch, acc, total_loss / len(train_loader.dataset)))
            history[epoch, 0] = acc
            history[epoch, 1] = total_loss / len(train_loader.dataset)

            if epoch >= args.start_correct and (epoch-5) % 10 == 0:
                changed_samples_ind, true_index, error_index, sharpened_probs = findAndCalibration(threshold, evidences_all, nearest_indices,similarity_matrix, train_index, train_Y, Y_true, args, classes_num, device, epoch)
                if len(error_index) != 0 and len(error_index) <= len(true_index):

                    random_numbers = torch.Tensor(np.random.beta(0.3, 0.3, size=true_index.shape[0])).to(device)
                    random_numbers = torch.max(random_numbers, 1-random_numbers)
                    train_Y_tensor = torch.Tensor(train_Y).to(device)
                    
                    y1 = train_Y_tensor[true_index]
                    y_mix = random_numbers.view(-1, 1) * sharpened_probs + (1 - random_numbers.view(-1, 1)) * y1
                    train_Y_tensor[error_index] = y_mix
                    for v in range(args.views):
                        if type(train_X[v]) == np.ndarray:
                            train_X[v] = torch.Tensor(train_X[v]).to(device)
                        x1 = train_X[v][error_index]
                        x2 = train_X[v][true_index]  
                        train_X[v][error_index] = random_numbers.view(-1, 1) * x1 + (1 - random_numbers.view(-1, 1)) * x2
                        train_X[v] = train_X[v].cpu().numpy()

                    # Create a new training set
                    dataset = MultiViewDataset(args.dataset, train_X, train_Y_origin, train_Y_tensor.detach().cpu().numpy(), Y_true)
                    train_dataset = Subset(dataset, train_index)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    total_error_index = error_index if total_error_index is None else torch.unique(torch.cat((total_error_index, error_index)))

                    train_Y = train_Y_tensor.detach().cpu().numpy()
                    threshold = threshold * 1.05

        # print("clean_model train time:", round(time.time() - start_time, 2), "s")
        return history[:,0].max()

    repeat_num = 5
    test_acc_loss_history = np.zeros([6, repeat_num])
    for repeat in range(repeat_num):  # Control total number of runs
        for i, ratio in enumerate(noise_ratios):
            print("======================================\nCurrent noise ratio: {}".format(ratio))
            args = loadConfig(args, ratio)
            test_acc_loss_history[i, repeat] = experiment(ratio, repeat, args.threshold)
    return test_acc_loss_history

def import_and_load_data(mu):
    # load data
    # with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/X.pkl", 'rb') as file:
    with open(args.data_path + '/' + "IDN" + "-0.00" + "/X.pkl", 'rb') as file:
        X = pickle.load(file)
    with open(args.data_path + '/' + "IDN" + "-{:.2f}".format(mu) + "/Y_true.pkl", 'rb') as file:
        Y_true = pickle.load(file)
    with open(args.data_path + '/' + "IDN" + "-{:.2f}".format(mu) + "/Y_noisy.pkl", 'rb') as file:
        Y_noisy = pickle.load(file)

    dataset = MultiViewDataset(args.dataset, X, Y_noisy, Y_noisy, Y_true)
    samples_num = Y_noisy.shape[0]
    classes_num = Y_noisy.shape[1]
    index = np.arange(samples_num)
    np.random.shuffle(index)
    train_index, test_index = index[:int(samples_num * 0.8)], index[int(samples_num * 0.8):]
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_index), shuffle=False)

    train_X = {}
    for v in range(args.views):
        train_X[v] = np.full_like(X[v], 0)
        train_X[v][train_index] = X[v][train_index]  # Here the test set is now equal to 0, but the length of the whole X is still the length of the whole dataset, in order to match the training process afterward.
    # train_Y = Y_noisy[train_index]
    return train_loader, test_loader, classes_num, train_X, Y_noisy, train_index, samples_num, Y_true


def test(model, test_loader, epoch):
    model.eval()
    correct_num, data_num = 0, 0
    # error_index = np.array([])
    for X, Y, indexes, corrected_Y, clean_Y in test_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        data_num += Y.size(0)
        with torch.no_grad():
            Y = Y.long().to(device)
            clean_Y = clean_Y.to(device)      
            _, evidence_a, _, _ ,_ = model(X, Y, corrected_Y, indexes, 0, 0, 0, None, args.l2)
            predicted = torch.argmax(evidence_a, dim=1)
            clean_Y_ind = torch.argmax(clean_Y, dim=1)
            correct_num += (predicted == clean_Y_ind).sum().item()
    return correct_num / data_num


if __name__ == "__main__":
    test_acc_loss_history = main()

    print("Noise\tAccuracy")
    for noise, acc in zip(noise_ratios, test_acc_loss_history.mean(axis=1)*100):
        print(f"{noise:.1f}\t\t{acc:.2f}%")
