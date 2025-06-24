import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import average_precision_score
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

import matplotlib.pyplot as plt
from lion_pytorch import Lion
from argparse import ArgumentParser

from transformer_model import *
from model import ResnetEncoder
from Myloader import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def evaluate(models, data_loader):
    
    encoder_head, encoder_tail, encoder_all = models
    encoder_head.eval()
    encoder_tail.eval()
    encoder_all.eval()
    running_loss, total, counter = 0.0, 0, 0
    with torch.no_grad():
        record_target_label = torch.zeros(1, 19).to(device)
        record_predict_label = torch.zeros(1, 19).to(device)
        
        for (imgs, labels, dicoms) in data_loader:
            counter += 1
            if counter % 100 == 0:
                print(f"{counter}th iter in loader...")

            imgs = imgs.to(device)
            labels = labels.to(device).squeeze(-1)
            
            outputs_head = torch.sigmoid(encoder_head(imgs))
            outputs_tail = torch.sigmoid(encoder_tail(imgs))
            outputs_all = torch.sigmoid(encoder_all(imgs))
            outputs = torch.cat((outputs_head, outputs_tail[:, 1:]), 1)
            outputs = (outputs + outputs_all) / 2.
            outputs[:, 8] = (outputs_all[:, 8] + outputs_head[:, 8] + outputs_tail[:, 8]) / 3.
                       
            record_target_label = torch.cat((record_target_label, labels), 0)
            record_predict_label = torch.cat((record_predict_label, outputs), 0)
            
        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]
        
        metric = MultilabelAveragePrecision(num_labels=19, average="macro")
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))
        
        metric = MultilabelAveragePrecision(num_labels=19, average="none")
        mAPs = metric(record_predict_label, record_target_label.to(torch.int32))

    return mAP, mAPs, record_target_label, record_predict_label


if __name__ == "__main__":
    
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = f"C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/test_ablation"
    weight_dir = "C:/Users/112062522/Downloads/112062522_whuang/research/hc/Datasets/results/checkpoint"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    batch_size = 32
    paths = [f"data/MICCAI_long_tail_{mode}.tfrecords" for mode in ['train', 'val', 'test']]
    index = [f"data/MICCAI_long_tail_{mode}.tfindex" for mode in ['train', 'val', 'test']]
    loaders = [Myloader_ensemble(p, i, batch_size, num_workers=0, image_size=256, shuffle=False) for p, i in zip(paths, index)]

    criterion = nn.BCEWithLogitsLoss()

    # head    
    checkpoint = torch.load(f"{weight_dir}/head_final_model_best.pt")      
    encoder_head = transformer_model(num_classes=9).to(device)
    encoder_head.load_state_dict(checkpoint['model_state_dict'])
 
    # tail
    checkpoint = torch.load(f"{weight_dir}/tail_final_model_best.pt")          
    encoder_tail = transformer_model(num_classes=11).to(device)
    encoder_tail.load_state_dict(checkpoint['model_state_dict'])

    # all
    checkpoint = torch.load(f"{weight_dir}/all_final_model_best.pt")       
    encoder_all = transformer_model(num_classes=19).to(device)
    encoder_all.load_state_dict(checkpoint['model_state_dict'])

    label_name = ['Lung Opacity','Cardiomegaly','Pleural Effusion','Atelectasis','No Finding','Pneumonia','Edema','Enlarged Cardiomediastinum','Support Devices','Consolidation','Pneumothorax','Fracture','Calcification of the Aorta','Tortuous Aorta','Subcutaneous Emphysema','Lung Lesion','Pneumomediastinum','Pneumoperitoneum','Pleural Other']

    for mode, loader in zip(['train', 'val', 'test'], loaders):
        if mode == 'train':
            continue
        mAP, mAPs, record_target_label, record_predict_label = evaluate([encoder_head, encoder_tail, encoder_all], loader)
        print(f"{mode} mAP: {mAP}")
        print(f"{mode} mAPs: {mAPs}")                
                
        results = {'record_target_label': record_target_label, 'record_predict_label': record_predict_label}
        np.save(output_path+f'/results_{mode}.npy', results)
        
