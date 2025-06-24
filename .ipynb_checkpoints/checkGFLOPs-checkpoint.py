import torch
import torch.nn as nn
import torchprofile
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import timm
from transformer_model import *


flops = 0
for n_classes in [9,11,19]:
    encoder = transformer_model(num_classes=n_classes)
    dummy_input = torch.randn(1, 3, 256, 256)
    flops += torchprofile.profile_macs(encoder, dummy_input) / 1e9  # Convert to GFLOPs
print(f'LTCXNet Total GFLOPs: {flops:.3f}')
exit()


num_classes=19
#---
model = models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
dummy_input = torch.randn(1, 3, 256, 256)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'Resnet18 Total GFLOPs: {flops:.3f}')

#---
model = models.resnet50(weights='DEFAULT')
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
dummy_input = torch.randn(1, 3, 256, 256)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'Resnet50 Total GFLOPs: {flops:.3f}')

#---
model = models.densenet121(weights='DEFAULT')
model.classifier = nn.Linear(model.classifier.in_features, 19)
dummy_input = torch.randn(1, 3, 256, 256)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'dense121 Total GFLOPs: {flops:.3f}')

#---
model = models.densenet161(weights='DEFAULT')
model.classifier = nn.Linear(model.classifier.in_features, 19)
dummy_input = torch.randn(1, 3, 256, 256)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'dense161 Total GFLOPs: {flops:.3f}')

#---
model = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=19, pretrained=True)
dummy_input = torch.randn(1, 3, 256, 256)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'convnext Total GFLOPs: {flops:.3f}')

#---
model = models.vit_b_16(weights='DEFAULT')
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
dummy_input = torch.randn(1, 3, 224, 224)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'vit Total GFLOPs: {flops:.3f}')

#---
model = models.swin_b(weights='DEFAULT')
model.head = torch.nn.Linear(model.head.in_features, num_classes)
dummy_input = torch.randn(1, 3, 224, 224)
flops = torchprofile.profile_macs(model, dummy_input) / 1e9  # Convert to GFLOPs
print(f'swin_transformer Total GFLOPs: {flops:.3f}')
