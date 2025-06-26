import torch
import argparse
import numpy as np
from model import TMNRR
from tqdm import tqdm
import platform
from utils.helpers import get_data_loaders
from utils.util import set_seed
from findAndCalibration import findAndCalibration
import faiss
from sklearn.metrics import accuracy_score
from extractFeature import extract_feature
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
On the first run, load the noiseless dataset and set the shuffle of tran_loader to False to save a feature file, after which restart the program and use it normally.
在第一次运行时，加载无噪声的数据集，并设置tran_loader的shuffle为False， 以保存一个特征文件，之后重新启动程序并正常使用。
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='input batch size for training', default=48)
parser.add_argument('--epochs', type=int, help='number of epochs to train', default=20)
parser.add_argument('--cos_epochs', type=int, help='cosine learning rate decay per epoch', default=10)
parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
parser.add_argument('--lr_T', type=float, help='learning rate', default=5e-5)
parser.add_argument('--dataset', type=str, help='Food101', default="Food101")
parser.add_argument('--IDN', type=int, help='noise ratio:0,10,20,...,70', default=0)
parser.add_argument('--warmUp', type=int, default=0)  
parser.add_argument('--start_correct', type=int, default=3)    
parser.add_argument('--patience', type=int, default=3)  
args = parser.parse_args()
saveAcc = False

args.dev = 0  # 0/1 development model
noise_ratios = np.array([0.5])  
args.k = 10
args.lamb = 0.01 
args.gamma = 10000
args.threshold = 0.95

if platform.system() == "Linux":
    args.data_path = "./Multi-modal_Released/datasets"
    args.cache_path = None
    args.num_workers = 12
    args.persistent_workers = True 
else:
    args.data_path = "./datasets"
    args.cache_path = "./cache"
    args.num_workers = 0 
    args.persistent_workers = False


args.dims = [[2048], [768]]  # Image and text dimensions
args.views = len(args.dims)
classes_num = 101

def experiment(ratio, repeat_num, threshold):
    print("====================================================")
    args.IDN = int(ratio * 100)  
    if args.IDN == 0:
        threshold = 0.99
    print('lr:', args.lr,'lr_T:', args.lr_T,'IDN:',args.IDN, 'warmup:',args.warmUp,'correct:',args.start_correct,'threshold:',threshold, 'k:',args.k)
    train_loader, val_loader, test_loader = get_data_loaders(args, True)  # todo Use shuffle=False only when saving features for the first time

    # Noise-free labeling in origins.
    train_features, origin_train_labels, sample_num = extract_feature(args, train_loader, args.dims[0][-1], args.dims[1][-1],device)
    origin_train_labels = torch.Tensor(origin_train_labels).to(device).long()
    k_batch_size = 2000
    # initialize the storage structure
    nearest_indices = np.zeros((args.views, sample_num, args.k), dtype=np.int64)
    nearest_similarities = np.zeros((args.views, sample_num, args.k))
    

    for v in range(args.views):
        nearest_indices[v], nearest_similarities[v] = compute_k_nearest_neighbors(sample_num, train_features[v], args.k, k_batch_size)
    print("Completion of k-nearest neighbor calculation")
    
    """ 
    training
    """
    history = np.zeros((args.epochs, 2))
    model = TMNRR(args, 101, 2, device, nearest_similarities, nearest_indices).to(device)

    optimizer1 = torch.optim.Adam([{'params': model.txtclf.parameters()},{'params': model.imgclf.parameters()}],
                                   lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.cos_epochs,eta_min=1e-7)  
    optimizer2 = torch.optim.Adam([model.matrixes], lr=args.lr_T, weight_decay=1e-5)
    

    corrected_labels = {}
    best_acc = 0.0
    total_error_index = None
    labels_all = torch.zeros(sample_num).to(device).long()  
    gt_all = torch.zeros(sample_num).to(device).long()  
    for epoch in range(0, args.epochs):
        model.train()
        total_loss = 0
        train_losses = []
        evidences_all = torch.zeros(args.views, sample_num, classes_num).to(device)
        accuracies = np.zeros((args.views)) 
        
        for batch in tqdm(train_loader, total=len(train_loader), bar_format="{l_bar}{bar:30}{r_bar}"):
            txt, segment, mask, img, tgt, idx, label = batch
            txt, img = txt.to(device), img.to(device)
            mask, segment = mask.to(device), segment.to(device)
            tgt = tgt.to(device)
            label = label.to(device)
            idx = idx.long().to(device)

            corrected_Y = label.clone()  # Copy tgt to avoid directly modifying the original tag

            for i, sample_idx in enumerate(idx):
                if sample_idx.item() in corrected_labels:
                    corrected_Y[i] = corrected_labels[sample_idx.item()]

            evidence, t_evidence, t_alpha, t_alpha_a, loss, _ = \
                model(args, txt, mask,segment,img,idx, label, corrected_Y,epoch,total_error_index)
                   
                    
            train_losses.append(loss.item())
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            if epoch >= args.warmUp:  # note warm-start
                optimizer2.step()

            t = model.matrixes.detach()
            t = torch.clamp(t, min=0)
            normalized_matrixes = t / t.sum(dim=-1, keepdim=True)
            with torch.no_grad():
                model.matrixes.data = normalized_matrixes
            for v_num in range(2):
                evidences_all[v_num][idx] = evidence[v_num]
            labels_all[idx] = label
            gt_all[idx] = tgt
        scheduler.step()
        for v in range(args.views):
            predicted_labels = evidences_all[v].argmax(dim=1)
            accuracy2 = (predicted_labels == labels_all).float().mean().item()
            accuracies[v] = accuracy2
        print(f"Training accuracy: {accuracies}")
        
        
        if epoch >= args.start_correct: 
            train_index = torch.arange(sample_num)  # Use the index of all samples

            train_Y = F.one_hot(labels_all, num_classes=classes_num).to(device)   
            # For samples with corrected labels, the corrected labels are used as train_Y, corrected_labels are of one-hot form
            for idx in corrected_labels:
                train_Y[idx] = F.one_hot(corrected_labels[idx], classes_num)  

            changed_samples_ind, true_index, error_index, changed_label = \
                findAndCalibration(threshold, evidences_all, nearest_indices, nearest_similarities, train_index, train_Y, args, classes_num, device, epoch,accuracies)
            
            if len(changed_samples_ind) != 0:
                for i, idx in enumerate(changed_samples_ind):
                    corrected_labels[idx.item()] = changed_label[i]
                
                origin_diff_mask = torch.ne(origin_train_labels, labels_all) 
                num_diff = torch.sum(origin_diff_mask).item()

                after_correct_noise_num = 0
                for idx, label in corrected_labels.items():
                    if label != origin_train_labels[idx]:
                        after_correct_noise_num += 1
                total_error_index = error_index if total_error_index is None else torch.unique(torch.cat((total_error_index, error_index)))
                threshold = threshold * 1.05
     
        metrics = test(model, test_loader, epoch)
        print('Epoch {} ====> test_acc: {:.4f}, loss = {}'.format(epoch, metrics['acc'], metrics['loss']))
        history[epoch, 0] = metrics['acc']
        history[epoch, 1] = total_loss / len(train_loader.dataset)
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early stopping, best acc: {:.4f}".format(best_acc))
                break

    return history[:,0].max()


def main():
    # global args, threshold
    repeat_num = 4
    test_acc_loss_history = np.zeros([6, repeat_num])
    for repeat in range(repeat_num):  # Control total number of runs
        for i, ratio in enumerate(noise_ratios):
            test_acc_loss_history[i, repeat] = experiment(ratio, repeat, args.threshold)

            cuda_device_idx = int(str(device).split(':')[-1]) if 'cuda' in str(device) else 0
            torch.cuda.set_device(cuda_device_idx)

            torch.cuda.empty_cache()
    return test_acc_loss_history

def compute_k_nearest_neighbors(sample_num, features, k, batch_size=4000):
    """
    Only the k nearest neighbors of each sample and their similarity are calculated and returned
    Args: 
    features: feature matrix, in the shape of [sample_num, feature_dim] 
    k: number of retained nearest neighbors 
    batch_size: batch size
        
    Returns: 
    nearest_indices: indexes of k nearest neighbors of each sample, in the shape of [sample_num, k] 
    nearest_similarities: similarity of each sample to its k nearest neighbors, in the shape of [sample_num, k]
    """

    features_np = features.astype(np.float32)
    cuda_device_idx = int(str(device).split(':')[-1]) if 'cuda' in str(device) else 0
        
    if platform.system() == "Linux":
        # Create GPU resources
        res = faiss.StandardGpuResources()
        # Creates a GPU device configuration object and specifies a GPU index
        gpu_config = faiss.GpuIndexFlatConfig()
        gpu_config.device = cuda_device_idx 
        d = features_np.shape[1]  
        index = faiss.GpuIndexFlatL2(res, d, gpu_config) 
    else:
        # Using the CPU version of faiss
        d = features_np.shape[1]  
        index = faiss.IndexFlatL2(d)  
    
    index.add(features_np)
    # Since faiss looks up itself, we need to look up k+1 nearest neighbors and then eliminate the first one (itself)
    distances = np.zeros((sample_num, k), dtype=np.float32)
    nearest_indices = np.zeros((sample_num, k), dtype=np.int64)
    # Batching queries to reduce memory usage
    for i in range(0, sample_num, batch_size):
        end_i = min(i + batch_size, sample_num)
        batch_distances, batch_indices = index.search(features_np[i:end_i], k+1)

        distances[i:end_i] = batch_distances[:, 1:k+1]
        nearest_indices[i:end_i] = batch_indices[:, 1:k+1]
    # start = time.time()
    # Compute adaptive sigma values for each sample
    sigma = np.mean(distances, axis=1, keepdims=True)  # Each sample's sigma = 1/k * sum of its k nearest neighbor distances

    sigma = np.maximum(sigma, 1e-5)
    nearest_similarities = np.exp(-distances / (sigma ** 2))

    return nearest_indices, nearest_similarities

def test(model, test_loader, epoch):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            txt, segment, mask, img, tgt, idx, label = batch
            txt, img = txt.to(device), img.to(device)
            mask, segment = mask.to(device), segment.to(device)
            tgt = tgt.to(device)
            label = label.to(device)
            
            corrected_Y = label.clone()  
            _, _, _, _, loss, evidence_a = \
                    model(args, txt, mask,segment,img,idx, label, corrected_Y, epoch)
            losses.append(loss.item())

            pred = evidence_a.argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            label = label.cpu().detach().numpy()
            labels.append(label)
    
    metrics = {"loss": np.mean(losses)}
   
    labels = [l for sl in labels for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(labels, preds)
    return metrics

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    set_seed(123)

    test_acc_loss_history = main()
    
    for i, ratio in enumerate(noise_ratios):
        max_acc = np.max(test_acc_loss_history[i, 0, :, 0])
        print(f"noise ratio {ratio:.2f}: Test Accuracy = {max_acc:.4f}")