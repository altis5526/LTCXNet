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
from sklearn.metrics import f1_score


if __name__ == "__main__":
    

    output_path = f"C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/test_ablation"     
    
    val_results = np.load(output_path+'/results_val.npy', allow_pickle=True).item()
    test_results = np.load(output_path+'/results_test.npy', allow_pickle=True).item()
    
    threshold = 0.5    
    val_pred = val_results['record_predict_label'].cpu().numpy()
    val_y = val_results['record_target_label'].cpu().numpy()
    val_pred = (val_pred >= threshold).astype(int)
    test_pred = test_results['record_predict_label'].cpu().numpy()
    test_y = test_results['record_target_label'].cpu().numpy()
    test_pred = (test_pred >= threshold).astype(int)
        
    val_mf1 = f1_score(val_y, val_pred, average='macro')
    test_mf1 = f1_score(test_y, test_pred, average='macro')
    print(f"val mf1: {val_mf1}")
    print(f"test mf1: {test_mf1}")
    
    f1_metric = MultilabelF1Score(num_labels=19, average="macro")
    val_mf1 = f1_metric(val_results['record_predict_label'], val_results['record_target_label'])
    test_mf1 = f1_metric(test_results['record_predict_label'], test_results['record_target_label'])
    print(f"val mf1: {val_mf1}")
    print(f"test mf1: {test_mf1}")
    
