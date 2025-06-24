import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn import metrics
import pandas as pd
import os
from argparse import ArgumentParser


def computeAUCratio_v3(test_record, sub_test_record, group_names, model_name, mask):
    
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
    
    path = "PATH/hc3_extra/fairness/results/0228_v9" 
    with open(f'{path}/{args.dataSeed}_record.pkl', 'rb') as file:
        record = pickle.load(file)
        
    output_folder = 'PATH/fairness/results/LTCXNet'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    group_names_dict = {
        'race': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN', 'OTHER'],
        'age': ['0-20', '20-40', '40-60', '60-80', '80-100'],
        'gender': ['F', 'M']
    }   
       
      
    # for mask_name in ['all', 'head', 'tail']:
    for mask_name in ['all']:
        if mask_name == 'all':
            mask = [True]*19
        elif mask_name == 'head':
            mask = [True]*9 + [False]*10
        elif mask_name == 'tail':
            mask = [False]*9 + [True]*10
        # mask[12], mask[13] = False, False
        mask[17] =  False
        # mask[18] =  False
        # mask[16] =  False

        results = np.zeros((4,2))
        for i, model_name in enumerate(record['test'].keys()):
            # for j, fairness_type in enumerate(['race', 'age', 'gender']):
            for j, fairness_type in enumerate(['race']):
                _ = computeAUCratio_v3(record['val'][model_name]['all'], record['test'][model_name][fairness_type], group_names_dict[fairness_type], model_name, mask)
                auc_ratio = np.mean(_)
                print(model_name, fairness_type, auc_ratio)
                results[i][j] = auc_ratio

        
        print('-------------------')
        
       
        
        df = pd.DataFrame(results, index=['mAUCratio convnext', 'mAUCratio mldecoder', 'mAUCratio aug', 'mAUCratio ensemble'], columns=['race', 'race std'])
        csv_file_path = f'{output_folder}/{args.dataSeed}_LTCXNet_{mask_name}.csv'
        df.to_csv(csv_file_path)


    
   