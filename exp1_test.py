import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import average_precision_score
from torchmetrics.classification import MultilabelAveragePrecision

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


def evaluate(model, loader):
    model.eval()
    test_running_loss = 0.0
    test_total = 0

    with torch.no_grad():
        record_target_label = torch.zeros(1, 19).to(device)
        record_predict_label = torch.zeros(1, 19).to(device)
        for (test_imgs, test_labels, test_dicoms) in loader:
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            test_output = model(test_imgs)
            loss = criterion(test_output, test_labels)

            test_running_loss += loss.item() * test_imgs.size(0)
            test_total += test_imgs.size(0)

            record_target_label = torch.cat((record_target_label, test_labels), 0)
            record_predict_label = torch.cat((record_predict_label, test_output), 0)


        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]

        metric = MultilabelAveragePrecision(num_labels=19, average="macro", thresholds=None)
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))

        metric = MultilabelAveragePrecision(num_labels=19, average="none")
        mAPs = metric(record_predict_label, record_target_label.to(torch.int32))
        
    return mAP, mAPs, test_running_loss, test_total

def arg_parser(parser):

  parser.add_argument("--seed", type=int)
  parser.add_argument("--model", type=str)
  
  args, unknown = parser.parse_known_args()

  return args, unknown

if __name__ == "__main__":
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)
    
    set_seed(args.seed)
    torch.cuda.set_device(0)

    batch_size = 20
    num_classes = 19
    num_workers = 0
    pin_memory = True

    test_path = "data/MICCAI_long_tail_test.tfrecords"
    test_index = "data/MICCAI_long_tail_test.tfindex"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == "resnet18":
        encoder = models.resnet18(weights='DEFAULT').to(device)
        encoder.fc = torch.nn.Linear(encoder.fc.in_features, num_classes).to(device)        
    elif args.model == "resnet50":
        encoder = models.resnet50(weights='DEFAULT').to(device)
        encoder.fc = torch.nn.Linear(encoder.fc.in_features, num_classes).to(device)
    elif args.model == "densenet121":
        encoder = models.densenet121(weights='DEFAULT').to(device)
        encoder.classifier = nn.Linear(encoder.classifier.in_features, num_classes).to(device)
    elif args.model == "densenet161":
        encoder = models.densenet161(weights='DEFAULT').to(device)
        encoder.classifier = nn.Linear(encoder.classifier.in_features, num_classes).to(device)
    elif args.model == "convnext":
        encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=num_classes, pretrained=True).to(device)
    elif args.model == "vit":
        encoder = models.vit_b_16(weights='DEFAULT').to(device)
        encoder.heads.head = torch.nn.Linear(encoder.heads.head.in_features, num_classes).to(device)
    elif args.model == "swin_transformer":
        encoder = models.swin_b(weights='DEFAULT').to(device)
        encoder.head = torch.nn.Linear(encoder.head.in_features, num_classes).to(device)
    
    # load model
    checkpoint = torch.load(f"C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/exp1/{args.seed}_{args.model}/model_best.pt")
    encoder.load_state_dict(checkpoint['model_state_dict'])
    
    if args.model == "vit" or args.model == "swin_transformer":
        image_size = 224
    else:
        image_size = 256
    test_loader = Myloader_ensemble(test_path, test_index, batch_size, num_workers=num_workers, image_size=image_size, shuffle=False, pin_memory=pin_memory)

    criterion = nn.BCEWithLogitsLoss()
    
    mAP, mAPs, test_running_loss, test_total = evaluate(encoder, test_loader)

    with open('C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/exp1_test_result.txt', 'a') as file:
        print(f"{args.seed}_{args.model} test mAP: {mAP}")
        file.write(f"{args.seed}_{args.model} test mAP: {mAP}")
        file.write('\n')