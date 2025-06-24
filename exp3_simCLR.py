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

import timm
import torch.nn.functional as F

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

def evaluate_simclr(model, loader):
    model.eval()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for img1, img2 in loader:
            img1, img2 = img1.to(device), img2.to(device)

            out1 = model(img1)
            out2 = model(img2)
            loss = criterion(out1, out2)

            running_loss += loss.item()
            total += img1.size(0)
                    
    return running_loss / total
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        """
        Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.
        
        Args:
            batch_size (int): Number of samples per batch.
            temperature (float): Controls sharpness of similarity distribution (default=0.5).
            device (str): Device to use ('cuda' or 'cpu').
        """
        super(NTXentLoss, self).__init__()
        self.temperature = max(0.1, min(1.0, temperature))  # Ensure temperature is reasonable
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss.
        
        Args:
            z_i (Tensor): First augmented batch of embeddings, shape (batch_size, feature_dim).
            z_j (Tensor): Second augmented batch of embeddings, shape (batch_size, feature_dim).

        Returns:
            Tensor: NT-Xent loss value.
        """
        # Step 1: Normalize embeddings (Prevents extreme similarity values)
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        batch_size = z_i.size(0)

        # Step 2: Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2N, D)

        # Step 3: Compute cosine similarity matrix
        sim_matrix = torch.mm(z, z.T)  # Shape: (2N, 2N)
        
        # Step 4: Remove self-similarity using fill_diagonal_()
        sim_matrix.fill_diagonal_(-5.0)  # Use -5 instead of -inf to prevent log issues

        # Step 5: Clamp similarity values to avoid extreme negatives
        sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

        # Step 6: Apply temperature scaling and compute log-softmax
        logits = sim_matrix / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)

        # Step 7: Construct target labels (positive pairs)
        labels = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels, labels], dim=0)  # Targets: (2N,)

        # Step 8: Compute cross-entropy loss
        loss = self.criterion(log_probs, labels)
        
        # print("Max similarity:", sim_matrix.max().item())
        # print("Min similarity:", sim_matrix.min().item())
        # print("Any NaN in embeddings?", torch.isnan(z_i).any().item(), torch.isnan(z_j).any().item())
        # print("Loss:", loss.item())
        
        return loss


def arg_parser(parser):

  parser.add_argument("--seed", type=int)
  
  args, unknown = parser.parse_known_args()

  return args, unknown

if __name__ == "__main__":
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)
    
    set_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = f"C:/research/hc3/experiments/simcLR"
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    epochs = 10
    batch_size = 25

    train_path = "data/MICCAI_long_tail_train.tfrecords"
    train_index = "data/MICCAI_long_tail_train.tfindex"
    val_path = "data/MICCAI_long_tail_val.tfrecords"
    val_index = "data/MICCAI_long_tail_val.tfindex"
    test_path = "data/MICCAI_long_tail_test.tfrecords"
    test_index = "data/MICCAI_long_tail_test.tfindex"

    opt_lr = 3e-5
    weight_decay = 1e-5

    num_classes = 19
        
    # encoder = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=128, pretrained=True).to(device)
    
    # criterion = NTXentLoss().to(device)

    # opt = Lion(encoder.parameters(), lr=opt_lr / 5, weight_decay = weight_decay * 5)
    
    # train_loader = Myloader_simclr_ensemble(train_path, train_index, batch_size, num_workers=0, shuffle=True, data_aug=True, image_size=256)
    # val_loader = Myloader_simclr_ensemble(val_path, val_index, batch_size, num_workers=0, shuffle=False, data_aug=False, image_size=256)
    
    # min_val_loss = 0
    # total = 0
    # scaler = torch.cuda.amp.GradScaler()

    # for epoch in range(epochs):
    #     encoder.train()
    #     running_loss = 0.0
    #     start_time = time.time()
    #     time_pin = time.time()
    #     count = 0
    #     logs = []

    #     for img1, img2 in train_loader:
    #         encoder.zero_grad()
    #         opt.zero_grad()

    #         img1, img2 = img1.to(device), img2.to(device)
                        
    #         with torch.autocast(device_type='cuda', dtype=torch.float16):
    #             out1 = encoder(img1)
    #             out2 = encoder(img2)
    #             loss = criterion(out1, out2)

    #         scaler.scale(loss).backward()
    #         scaler.step(opt)
    #         scaler.update()

    #         running_loss += loss.item()
    #         count += img1.size(0)

    #         if count % 1000 == 0:
    #             if total == 0:
    #                 print(f"epoch {epoch}: {count}/unknown | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
    #                 logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
    #             elif total != 0:
    #                 print(f"epoch {epoch}: {count}/{total} {count/total*100:.2f}% | train loss: {running_loss / count:.4f} | duration: {time.time() - time_pin:.2f} seconds")
    #                 logs.append("epoch "+ str(epoch) +": "+ str(count) +"/unknown | train loss: "+ str(round(running_loss / count, 4)) +" | duration: "+ str(round(time.time() - time_pin)) +" seconds")
    #             time_pin = time.time()
                
    #     total = count
    #     val_loss = evaluate_simclr(encoder, val_loader)        
        
    #     if val_loss > min_val_loss:
    #         min_val_loss = val_loss
    #         torch.save({
    #             'model_state_dict': encoder.state_dict(),
    #             'optimizer_state_dict': opt.state_dict(),
    #         }, f"{output_path}/backbone_best.pt")

    #     end_time = time.time()
    #     duration = end_time - start_time
        
    #     print(f"epoch {epoch}: validation loss: {val_loss} | duration: {duration}")
        
    #     with open(output_path+f'/output.txt', 'a') as file:
    #         for log in logs:
    #             file.write(f"{log}\n")
    #         file.write(f"epoch {epoch}: validation loss: {val_loss} | duration: {duration}")

    #     with open(output_path+f'/result.txt', 'a') as file:
    #         file.write(f"epoch {epoch}: validation loss: {val_loss} | duration: {duration}\n")
                

    # ##----------------
    # # finetune
    # del train_loader
    # del val_loader
    # del encoder
    
    batch_size = 32
    
    encoder = transformer_model(num_classes).to(device)
    
    state_dict = torch.load(f"{output_path}/backbone_best.pt")['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
    encoder.model.load_state_dict(state_dict, strict=False)
    
    opt = Lion(encoder.parameters(), lr=opt_lr / 5, weight_decay = weight_decay * 5)
    
    train_loader = Myloader_ensemble(train_path, train_index, batch_size, num_workers=0, shuffle=True, data_aug=True, image_size=256)
    val_loader = Myloader_ensemble(val_path, val_index, batch_size, num_workers=0, shuffle=False, data_aug=False, image_size=256)

    criterion = nn.BCEWithLogitsLoss()
        
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
        
        