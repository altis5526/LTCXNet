import torch
from transformer_model import transformer_model, transformer_model_LWS
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from Myloader import *
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import f1_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def evaluate(encoder, data_loader):
    
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
            
            outputs = torch.sigmoid(encoder(imgs))                      
            record_target_label = torch.cat((record_target_label, labels), 0)
            record_predict_label = torch.cat((record_predict_label, outputs), 0)
            
            
        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]
        
        metric = MultilabelAveragePrecision(num_labels=19, average="macro")
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))
        
        metric = MultilabelAveragePrecision(num_labels=19, average="none")
        mAPs = metric(record_predict_label, record_target_label.to(torch.int32))
        
        macro_f1 = f1_score(record_target_label.cpu().numpy(), record_predict_label.cpu().numpy() > 0.5, average='macro', zero_division=0.0)

    return mAP, mAPs, record_target_label, record_predict_label, macro_f1


def arg_parser(parser):

  parser.add_argument("--dataSeed", type=int)
  parser.add_argument("--model", type=str, default='tmp')
  
  args, unknown = parser.parse_known_args()

  return args, unknown


if __name__ == "__main__":
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)

    set_seed(123)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dir = f"weights/{args.model}"
    data_path = "data"
        
    batch_size = 32
    num_classes = 19

    paths = [f"{data_path}/MICCAI_Resample_{mode}_seed{args.dataSeed}.tfrecords" for mode in ['val', 'test']]
    index = [None, None]
    # index = [f"{data_path}/MICCAI_long_tail_{mode}.tfindex" for mode in ['val', 'test']]

    loaders = [Myloader_ensemble(p, i, batch_size, num_workers=0, shuffle=False) for p, i in zip(paths, index)]


    if args.model == 'cRT':
        checkpoint = torch.load(f"{weight_dir}/model_best_finetuned.pt")
        encoder = transformer_model(num_classes=19).to(device)
    elif args.model == 'LWS':
        checkpoint = torch.load(f"{weight_dir}/model_best_finetuned.pt")
        encoder = transformer_model_LWS(num_classes=num_classes).to(device)
    else:
        checkpoint = torch.load(f"{weight_dir}/model_best.pt")       
        encoder = transformer_model(num_classes=19).to(device)
    encoder.load_state_dict(checkpoint['model_state_dict'])

    label_name = ['Lung Opacity','Cardiomegaly','Pleural Effusion','Atelectasis','No Finding','Pneumonia','Edema','Enlarged Cardiomediastinum','Support Devices','Consolidation','Pneumothorax','Fracture','Calcification of the Aorta','Tortuous Aorta','Subcutaneous Emphysema','Lung Lesion','Pneumomediastinum','Pneumoperitoneum','Pleural Other']

    for mode, loader in zip(['val', 'test'], loaders):
        
        mAP, mAPs, record_target_label, record_predict_label, mf1 = evaluate(encoder, loader)
        
        np.save(f'experiments/tmp/{mode}-{args.model}-{args.dataSeed}.npy', {
            'y_true': record_target_label.cpu().numpy(),
            'y_pred': record_predict_label.cpu().numpy(),
        })
        
        with open(f'experiments/final_{mode}_result.txt', 'a') as file:
            file.write(f"{args.model}-{args.dataSeed} mAP: {mAP} | macro_f1: {mf1}\n")