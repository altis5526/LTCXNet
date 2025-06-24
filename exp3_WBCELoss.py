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

def evaluate(model, loader, num_classes):
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        record_target_label = torch.zeros(1, num_classes).to(device)
        record_predict_label = torch.zeros(1, num_classes).to(device)
        for (imgs, labels, dicoms) in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(-1)

            output = model(imgs)
            loss = criterion(output, labels)

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

            record_target_label = torch.cat((record_target_label, labels), 0)
            record_predict_label = torch.cat((record_predict_label, output), 0)

        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]

        metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro", thresholds=None)
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))

        metric = MultilabelAveragePrecision(num_labels=num_classes, average="none")
        mAPs = metric(record_predict_label, record_target_label.to(torch.int32))
        
    return mAP, mAPs, running_loss, total

def arg_parser(parser):

  parser.add_argument("--seed", type=int)
  parser.add_argument("--name", type=str, default='tmp')
  
  args, unknown = parser.parse_known_args()

  return args, unknown

if __name__ == "__main__":
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)
    
    set_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = f"C:/research/hc3/experiments/{args.name}"
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    epochs = 10
    batch_size = 32

    train_path = "data/MICCAI_long_tail_train.tfrecords"
    train_index = "data/MICCAI_long_tail_train.tfindex"
    val_path = "data/MICCAI_long_tail_val.tfrecords"
    val_index = "data/MICCAI_long_tail_val.tfindex"
    test_path = "data/MICCAI_long_tail_test.tfrecords"
    test_index = "data/MICCAI_long_tail_test.tfindex"

    opt_lr = 3e-5
    weight_decay = 1e-5

    num_classes = 19
        
    encoder = transformer_model(num_classes=num_classes).to(device)
    opt = Lion(encoder.parameters(), lr=opt_lr / 5, weight_decay = weight_decay * 5)
    
    train_loader = Myloader_ensemble(train_path, train_index, batch_size, num_workers=0, shuffle=True, data_aug=True, image_size=256)
    val_loader = Myloader_ensemble(val_path, val_index, batch_size, num_workers=0, shuffle=False, data_aug=False, image_size=256)

    pos_cnt = np.array([57591, 55020, 53029, 51055, 34376, 32131, 30553, 19880, 73639, 
                    12406, 11579, 7823, 2595, 2212, 1900, 1707, 571, 448, 373])
    total = 182380
    pos_weight = torch.tensor((total - pos_cnt) / pos_cnt, device=device)
    pos_weight = torch.clamp(pos_weight, min=0.1, max=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses = []
    val_losses = []

    max_map = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        encoder.train()
        running_loss = 0.0
        start_time = time.time()
        time_pin = time.time()
        count = 0
        logs = []

        for (imgs, labels, dicom_id) in train_loader:
            encoder.zero_grad()
            opt.zero_grad()

            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(-1)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = encoder(imgs)
                loss = criterion(output, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            if count % 1024 == 0:
                if total == 0:
                    print(f"epoch {epoch}: {count}/unknown | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
                    logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
                elif total != 0:
                    print(f"epoch {epoch}: {count}/{total} {count/total*100:.2f}% | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
                    logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
                time_pin = time.time()

        total = count
        mAP, mAPs, val_running_loss, val_total = evaluate(encoder, val_loader, num_classes)
        
        train_losses.append(running_loss / count)
        val_losses.append(val_running_loss)
        
        if mAP > max_map:
            max_map = mAP
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, f"{output_path}/model_best.pt")

        end_time = time.time()
        duration = end_time - start_time
        
        print(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {val_running_loss / val_total} | duration: {duration}")
        print(f"epoch {epoch}: validation mAPs: {mAPs}")
        
        with open(output_path+f'/output.txt', 'a') as file:
            for log in logs:
                file.write(f"{log}\n")
            file.write(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {val_running_loss / val_total} | duration: {duration}")

        with open(output_path+f'/result.txt', 'a') as file:
            file.write(f"epoch {epoch}: validation mAP: {mAP} | validation loss: {val_running_loss / val_total} | duration: {duration}\n")
            file.write(f"epoch {epoch}: validation mAPs: {mAPs}\n\n")
                

    ## testing
    # del train_loader
    # del val_loader
    # del encoder
    
    # encoder = transformer_model(num_classes=num_classes).to(device)
    # encoder.load_state_dict(torch.load(f"{output_path}/model_best.pt")['model_state_dict'])
    # encoder.eval()
    
    # test_loader = Myloader_ensemble(test_path, test_index, batch_size, num_workers=0, shuffle=False, data_aug=False, image_size=256)
    
    # mAP, mAPs, test_running_loss, test_total = evaluate(encoder, test_loader, num_classes)
    # print(f"test mAP: {mAP} | test loss: {test_running_loss / test_total}")
    # print(f"test mAPs: {mAPs}")
    
    # with open(output_path+f'/result.txt', 'a') as file:
    #     file.write(f"test mAP: {mAP} | test loss: {test_running_loss / test_total}\n")
    #     file.write(f"test mAPs: {mAPs}\n\n")