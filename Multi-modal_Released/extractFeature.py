import torch
import numpy as np
from tqdm import tqdm
import os
from encoder import ImageClf, BertClf


def extract_feature(args, train_loader, img_dim, text_dim, device):
    feature_dir = "./Multi-modal_Released/datasets/" 
    os.makedirs(feature_dir, exist_ok=True)
    if args.dev == 0:
        feature_file = os.path.join(feature_dir, f"features_views{args.views}.npz")
    else:
        feature_file = os.path.join(feature_dir, f"features_views{args.views}_dev.npz")
    sample_num = len(train_loader.dataset)

    # Check if cached features exist
    if os.path.exists(feature_file):
        print(f"Loading cached features from: {feature_file}")
        cached_data = np.load(feature_file, allow_pickle=True)
        train_features = {}
        train_features[0] = cached_data['train_features_0']
        train_features[1] = cached_data['train_features_1']
        train_labels = cached_data['train_labels']
        return train_features, train_labels, sample_num

    # Extract features if no cache exists
    print("No cache found, extracting features...")
    
    # Initialize encoders
    img_encoder = ImageClf(args).to(device)
    text_encoder = BertClf(args).to(device)
    if args.dev == 0:
        img_encoder.load_state_dict(torch.load("./Multi-modal_Released/checkpoint/train_img_encoder.pth"))  
        text_encoder.load_state_dict(torch.load("./Multi-modal_Released/checkpoint/train_text_encoder.pth"))
    else:
        img_encoder.load_state_dict(torch.load("./Multi-modal_Released/checkpoint/val_img_encoder.pth"))
        text_encoder.load_state_dict(torch.load("./Multi-modal_Released/checkpoint/val_text_encoder.pth"))

    # Initialize feature arrays
    img_features = np.zeros((sample_num, img_dim), dtype=np.float32)
    text_features = np.zeros((sample_num, text_dim), dtype=np.float32)
    train_labels = np.zeros(sample_num, dtype=np.int64)

    # Extract features
    img_encoder.eval()
    text_encoder.eval()

    print("Extracting training features...")
    with torch.no_grad():
        for batch in tqdm(train_loader, total=len(train_loader)):
            txt, segment, mask, img, tgt, idx, label = batch
            txt, img = txt.to(device), img.to(device)
            mask, segment = mask.to(device), segment.to(device)
            tgt = tgt.to(device)
            
            # Extract features
            _, txt_f = text_encoder(txt, mask, segment)
            _, img_f = img_encoder(img)
            
            # Store features and labels
            img_features[idx] = img_f.cpu().numpy()
            text_features[idx] = txt_f.cpu().numpy()
            train_labels[idx] = tgt.cpu().numpy()

    train_features = {0: img_features, 1: text_features}

    # Print feature shapes
    for v in range(args.views):
        print(f"View {v} feature shape: {train_features[v].shape}")
    print(f"Training labels shape: {train_labels.shape}")
    
    # Save features to cache
    print(f"Saving features to: {feature_file}")
    save_dict = {f'train_features_{v}': train_features[v] for v in range(args.views)}
    save_dict.update({'train_labels': train_labels})
    np.savez_compressed(feature_file, **save_dict)
    print("Feature extraction and caching completed!")
    
    return train_features, train_labels, sample_num