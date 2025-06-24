import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn import metrics
import pandas as pd
import os
from argparse import ArgumentParser

def computeEO_v3(test_record, sub_test_record, group_names, model_name, mask):
    
    fnrs = []
    aucs = []
    mAPs = []
    for group_id in sub_test_record.keys():
        pred, target = test_record[0]['predict'], test_record[0]['target']        
        test_pred, test_target = sub_test_record[group_id]['predict'], sub_test_record[group_id]['target']
        

        if model_name != 'ensemble':
            pred, test_pred = torch.sigmoid(pred), torch.sigmoid(test_pred)
        pred = np.array(pred[:, mask]).astype(float)
        target = np.array(target[:, mask]).astype(int)
        test_pred = np.array(test_pred[:, mask]).astype(float)
        test_target = np.array(test_target[:, mask]).astype(int)

        AUC = metrics.roc_auc_score(test_target, test_pred, average="macro") 
        aucs.append(AUC)

     
    min_auc = np.min(aucs)
    max_auc = np.max(aucs)
    print(f"auc ratio: {min_auc/max_auc}")
    print(aucs)

    return min_auc/max_auc


def arg_parser(parser):
  parser.add_argument("--dataSeed", type=int)  
  args, unknown = parser.parse_known_args()
  return args, unknown


if __name__ == '__main__':
    
    parser = ArgumentParser()
    args, unknown = arg_parser(parser)

    model_names = ['LDAMLoss', 'CBLoss', 'cRT', 'LWS', 'MLROS', 'ROS', 'simCLR', 'WBCELoss', 'bsoftmaxLoss', 'focalLoss']
    path = f"PATH/hc3_extra/fairness/results/0609_v10"
    

    for model in model_names:
        with open(f'{path}/{model}_{args.dataSeed}_record.pkl', 'rb') as file:
            record = pickle.load(file)
            
        output_folder = f'results/{model}'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        
        group_names_dict = {
            'race': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'],
        }   
        print(record['test'].keys())
           
          
        for mask_name in ['all']:
            if mask_name == 'all':
                mask = [True]*19
            elif mask_name == 'head':
                mask = [True]*9 + [False]*10
            elif mask_name == 'tail':
                mask = [False]*9 + [True]*10
            # mask[12], mask[13] = False, False
            mask[17] =  False
    
            results = np.zeros((1,1))
            for i, model_name in enumerate(record['test'].keys()):
                print(model_name)
                for j, fairness_type in enumerate(['race']):
                    _ = computeEO_v3(record['val'][model_name]['all'], record['test'][model_name][fairness_type], group_names_dict[fairness_type], model_name, mask)
                    auc_ratio = np.mean(_)
                    print(model_name, fairness_type, auc_ratio)
                    results[i][j] = auc_ratio
    
            
            print('-------------------')
            
            
            df = pd.DataFrame(results, index=['mAUC ratio'], columns=['race'])
            csv_file_path = f'{output_folder}/{args.dataSeed}_{model}_{mask_name}.csv'
            df.to_csv(csv_file_path)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # model_names = ['raw', 'aug', 'ensemble', 'byol']
    # mode = 'val'
    
    # record_raw = record[mode]['aug']
    # record_aug = record[mode]['raw']
    
    # for fairness_type in record_aug.keys():

    #     tmp0 = []
    #     tmp1 = []
    #     for group_id in record_aug[fairness_type].keys():
            
    #         if fairness_type != 'all':
    #             tmp0.append(record_aug[fairness_type][group_id]['mAPs'].numpy())
    #             tmp1.append(record_raw[fairness_type][group_id]['mAPs'].numpy())
        
    #     label_name1 = ['Atelectasis','Calcification of the Aorta','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Subcutaneous Emphysema','Support Devices','Tortuous Aorta']
        
    #     label_name2 = [
    #         'Pneumothorax',
    #         'Fracture',
    #         'Calcification of the Aorta',
    #         'Tortuous Aorta',
    #         'Subcutaneous Emphysema',
    #         'Lung Lesion',
    #         'Pneumomediastinum',
    #         'Pneumoperitoneum',
    #         'Pleural Other'
    #     ]
        
    #     idx = [label_name1.index(name) for name in label_name2]
    #     # print(idx)
    #     if fairness_type != 'all':
    #         tmp0 = np.array(tmp0)
    #         tmp_max = np.nanmax(tmp0, axis=0)
    #         tmp_min = np.nanmin(tmp0, axis=0)
    #         tmp0 = tmp_max - tmp_min
    #         score = np.mean(tmp0[idx])
    #         print(fairness_type, "with data aug", score)
            
    #         tmp1 = np.array(tmp1)
    #         tmp_max = np.nanmax(tmp1, axis=0)
    #         tmp_min = np.nanmin(tmp1, axis=0)
    #         tmp1 = tmp_max - tmp_min
    #         score = np.mean(tmp1[idx])
    #         print(fairness_type, "w/o data aug", score)
