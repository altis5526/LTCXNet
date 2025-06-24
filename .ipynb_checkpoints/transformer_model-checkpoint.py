import torch
import timm
import torch.nn as nn
from ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
import types


class backbone_model(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.model = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        x = self.model(x)    
        return x
    
class transformer_model(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.model = timm.create_model('convnext_small.fb_in22k_ft_in1k', num_classes=num_classes, pretrained=True)#'convnext_base.fb_in22k_ft_in1k_384'
        self.model.head = nn.Identity()
        self.pos_encoding = Summer(PositionalEncoding2D(768))#Summer(PositionalEncoding2D(1024))# 
        self.head = MLDecoder(num_classes=num_classes, initial_num_features=768)#MLDecoder(num_classes=19, initial_num_features=1024)# 

    def forward(self, x):
        x = self.model(x)           # (32, 3, 256, 256) -> (32, 768, 8, 8)
        x = self.pos_encoding(x)    # (32, 768, 8, 8) -> (32, 768, 8, 8)
        x = self.head(x)            # (32, 768, 8, 8) -> (32,19)
        return x

def LWS_call(self, h, duplicate_pooling, out_extrap):
    
    # h: [batch_size, num_queries, embed_dim]
    # duplicate_pooling: [num_queries, embed_dim, duplicate_factor]
    # out_extrap: [batch_size, num_queries, duplicate_factor]  
    # f: [num_queries]  
       
    for i in range(h.shape[1]):
        h_i = h[:, i, :]
        if len(duplicate_pooling.shape) == 3:
            w_i = duplicate_pooling[i, :, :]
        else:
            w_i = duplicate_pooling
            
        # normalize wi
        w_i = w_i.view(-1)      # [embed_dim * duplicate_factor]
        norm = torch.norm(w_i, p=2, dim=0)
        w_i = w_i / norm**self.f[i]
            
        out_extrap[:, i, :] = torch.matmul(h_i, w_i) + 1  # Custom change: Adding 1 to results
        
class transformer_model_LWS(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.model = transformer_model(num_classes=num_classes)

        # Modify GroupFC to include `f`
        self.model.head.decoder.group_fc.__class__.f = nn.Parameter(torch.randn(num_classes))
        self.model.head.decoder.group_fc.__call__ = types.MethodType(LWS_call, self.model.head.decoder.group_fc)

    def forward(self, x):
        x = self.model(x)           
        return x
   
class ensemble_model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_all = transformer_model(num_classes=19)
        self.model_head = transformer_model(num_classes=9)
        self.model_tail = transformer_model(num_classes=11)

    def forward(self, x):
        y_all = self.model_all(x)
        y_head = self.model_head(x)
        y_tail = self.model_tail(x)

        # get ensemble from the head and tail
        y_other = torch.cat((y_head[:, ], y_tail[:, 1:]), 1)

        # average the all and ensemble 
        y_pred = (y_all + y_other) / 2.

        # get avg for support device
        y_pred[:, 8] = (y_all[:, 8] + y_head[:, 8] + y_tail[:, 8]) / 3.

        return y_pred
 
# add activation for eval purpose
class ensemble_model_v2_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_all = transformer_model(num_classes=19)
        self.model_head = transformer_model(num_classes=9)
        self.model_tail = transformer_model(num_classes=11)
        
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y_all = self.activation((self.model_all(x)))
        y_head = self.activation((self.model_head(x)))
        y_tail = self.activation((self.model_tail(x)))

        # get ensemble from the head and tail
        y_other = torch.cat((y_head[:, ], y_tail[:, 1:]), 1)

        # average the all and ensemble 
        y_pred = (y_all + y_other) / 2.

        # get avg for support device
        y_pred[:, 8] = (y_all[:, 8] + y_head[:, 8] + y_tail[:, 8]) / 3.

        return y_pred
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # test = ensemble_model().to(device)
    # test = transformer_model()
    test = backbone_model()
    test.model.head = nn.Identity()
    print(test)

    # tens = torch.randn(2, 3, 256, 256).to(device)
    tens = torch.randn(2, 3, 256, 256)
    out = test(tens)
    print(out.shape)