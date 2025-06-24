import torch
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
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
import pickle

import sys
import os
sys.path.append(os.path.abspath(".."))
from transformer_model import *
from Myloader import *

from argparse import ArgumentParser

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def evaluate(model, data_loader, output_path, mode):
    
    model.eval()
    running_loss, total, counter = 0.0, 0, 0
    with torch.no_grad():
        # 5 groups, White, Black, Hispanic, Asian, Other
        # 5 age groups, 0-19, 20-39, 40-59, 60-79, 80-99
        # 2 groups, M, F
        record = {
            'all': {i: {'target': torch.zeros(1,19), 'predict': torch.zeros(1, 19), 'mAP': None, 'mAPs': None} for i in range(1)},
            'race': {i: {'target': torch.zeros(1,19), 'predict': torch.zeros(1, 19), 'mAP': None, 'mAPs': None} for i in range(5)},
            'age': {i: {'target': torch.zeros(1,19), 'predict': torch.zeros(1, 19), 'mAP': None, 'mAPs': None} for i in range(5)},
            'gender': {i: {'target': torch.zeros(1,19), 'predict': torch.zeros(1, 19), 'mAP': None, 'mAPs': None} for i in range(2)},
        }
        
        for (imgs, labels, dicoms, ages, genders, races) in data_loader:
            counter += 1
            if counter % 100 == 0:
                print(f"{counter}th iter in loader...")
                
            outputs = model(imgs.to(device)).to('cpu')
            labels = labels.squeeze(-1)
            
            record['all'][0]['target'] = torch.cat((record['all'][0]['target'], labels), 0)
            record['all'][0]['predict'] = torch.cat((record['all'][0]['predict'], outputs), 0)
            
            # race
            for label, output, race in zip(labels, outputs, races):
                if race.find('WHITE') != -1:
                    id = 0
                elif race.find('BLACK') != -1:
                    id = 1
                elif race.find('HISPANIC') != -1:
                    id = 2
                elif race.find('ASIAN') != -1:
                    id = 3
                else:
                    id = 4
                record['race'][id]['target'] = torch.cat((record['race'][id]['target'], label.reshape(1, -1)), 0)
                record['race'][id]['predict'] = torch.cat((record['race'][id]['predict'], output.reshape(1, -1)), 0)
            
            # age
            idx = [int(age / 20) for age in ages]
            for label, output, id in zip(labels, outputs, idx):
                id = np.clip(id, 0, 4)
                record['age'][id]['target'] = torch.cat((record['age'][id]['target'], label.reshape(1, -1)), 0)
                record['age'][id]['predict'] = torch.cat((record['age'][id]['predict'], output.reshape(1, -1)), 0)
                
            # gender
            for label, output, gender in zip(labels, outputs, genders):
                if gender == 'F':
                    record['gender'][0]['target'] = torch.cat((record['age'][0]['target'], label.reshape(1, -1)), 0)
                    record['gender'][0]['predict'] = torch.cat((record['age'][0]['predict'], output.reshape(1, -1)), 0)
                elif gender == 'M':
                    record['gender'][1]['target'] = torch.cat((record['age'][1]['target'], label.reshape(1, -1)), 0)
                    record['gender'][1]['predict'] = torch.cat((record['age'][1]['predict'], output.reshape(1, -1)), 0)
    
            # break
    
        # eval
        for fairness_type in record.keys():
            for key in record[fairness_type].keys():
                record[fairness_type][key]['predict'] = record[fairness_type][key]['predict'][1::]
                record[fairness_type][key]['target'] = record[fairness_type][key]['target'][1::]       

                if record[fairness_type][key]['target'].size(0) != 0:
                    metric = MultilabelAveragePrecision(num_labels=19, average="macro")
                    mAP = metric(record[fairness_type][key]['predict'], record[fairness_type][key]['target'].to(torch.int32))
                    record[fairness_type][key]['mAP'] = mAP
                
                    metric = MultilabelAveragePrecision(num_labels=19, average="none")
                    mAPs = metric(record[fairness_type][key]['predict'], record[fairness_type][key]['target'].to(torch.int32))
                    record[fairness_type][key]['mAPs'] = mAPs

    return record

def arg_parser(parser):
  parser.add_argument("--dataSeed", type=int)  
  args, unknown = parser.parse_known_args()
  return args, unknown

if __name__ == "__main__":
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)

    set_seed(123)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = "C:/research/hc3/fairness"
    output_path = f"{folder}/results/0228_v9"
    data_path = "../data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    batch_size = 10
    num_classes = 19

    model_names = ['convnext', 'mld', 'aug', 'ensemble']
    
    records = {'val': {}, 'test': {}}
    for i, model_name in enumerate(model_names):
        if model_name == 'convnext':      
            checkpoint = torch.load(f"{folder}/checkpoint/163_model_best.pt")
            encoder = backbone_model().to(device)
        elif model_name == 'mld':        
            checkpoint = torch.load(f"{folder}/checkpoint/all_no_aug_model_best.pt")     
            encoder = transformer_model().to(device)
        elif model_name == 'aug':        
            checkpoint = torch.load(f"{folder}/checkpoint/all_final_model_best.pt")     
            encoder = transformer_model().to(device)
        elif model_name == 'ensemble':   
            checkpoint_all = torch.load(f"{folder}/checkpoint/all_final_model_best.pt")
            checkpoint_head = torch.load(f"{folder}/checkpoint/head_final_model_best.pt")
            checkpoint_tail = torch.load(f"{folder}/checkpoint/tail_final_model_best.pt")
            encoder = ensemble_model_v2_5().to(device)
        
        if model_name in ['convnext', 'mld', 'aug']:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            encoder.model_all.load_state_dict(checkpoint_all['model_state_dict'])
            encoder.model_head.load_state_dict(checkpoint_head['model_state_dict'])
            encoder.model_tail.load_state_dict(checkpoint_tail['model_state_dict'])
        
        encoder.eval()
        
        for mode in ['val', 'test']:
            path = f"{data_path}/MICCAI_Resample_{mode}_seed{args.dataSeed}.tfrecords"
            index = None           
            loader = Myloader_fairness_ensemble(path, index, batch_size, num_workers=0, shuffle=False)
            records[mode][model_name] = evaluate(encoder, loader, output_path, mode=mode)
            
            with open(f'{output_path}/{args.dataSeed}_{model_name}_{mode}.pkl', 'wb') as file:
                pickle.dump(records[mode][model_name], file)

    with open(f'{output_path}/{args.dataSeed}_record.pkl', 'wb') as file:
        pickle.dump(records, file)
