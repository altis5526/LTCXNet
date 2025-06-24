import torch
import numpy as np
import os
import random
from Myloader import *
from transformer_model import *
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import average_precision_score
from sklearn.metrics import multilabel_confusion_matrix, f1_score
import pickle
    
    
def get_demographic(demographics, data_loader):
        
    for (imgs, labels, dicoms, ages, genders, races) in data_loader:
        
        # race
        for dicom, race, age, gender, label in zip(dicoms, races, ages, genders, labels):
            
            # print(dicom, race, age, gender)
            # print(ds)
            
            demographics['dicom'].append(dicom)
            
            if race.find('WHITE') != -1:
                demographics['race'].append('WHITE')
            elif race.find('BLACK') != -1:
                demographics['race'].append('BLACK')
            elif race.find('HISPANIC') != -1:
                demographics['race'].append('HISPANIC')
            elif race.find('ASIAN') != -1:
                demographics['race'].append('ASIAN')
            else:
                demographics['race'].append('OTHER')
                
            demographics['age'].append(age)
            demographics['gender'].append(gender)
            demographics['label'].append(label)

    return demographics


if __name__ == "__main__":


    output_path = f"experiments/demographics"
    data_path = "data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if not os.path.exists(f'{output_path}/demographics.npy'):
        demographics = {
            'dicom': [],
            'race': [],
            'age': [],
            'gender': [],
            'label': [],
        }
        for mode in ['train', 'val', 'test']:
            path = f"{data_path}/MICCAI_long_tail_{mode}.tfrecords"
            index = f"{data_path}/MICCAI_long_tail_{mode}.tfindex"
            
            loader = Myloader_fairness_ensemble(path, index, batch_size=100, num_workers=4, shuffle=False)
            get_demographic(demographics, loader)

        np.save(f'{output_path}/demographics.npy', demographics)


    
    demographics = np.load(f'{output_path}/demographics.npy', allow_pickle=True).item()
    
    # print(set(demographics['age']))
    # print(set(demographics['race']))
    # print(set(demographics['gender']))
    
    age_cnt = np.zeros(5)
    race_cnt = np.zeros(5)
    gender_cnt = np.zeros(2)
    i = 0
    for age, gender, race in zip(demographics['age'], demographics['gender'], demographics['race']):
        if age < 20 and age > 0:
            age_cnt[0] += 1
        elif age < 40:
            age_cnt[1] += 1
        elif age < 60:
            age_cnt[2] += 1
        elif age < 80:
            age_cnt[3] += 1
        elif age < 100:
            age_cnt[4] += 1
            
        if race == 'WHITE':
            race_cnt[0] += 1
        elif race == 'BLACK':
            race_cnt[1] += 1
        elif race == 'HISPANIC':
            race_cnt[2] += 1
        elif race == 'ASIAN':
            race_cnt[3] += 1
        elif race == 'OTHER':
            race_cnt[4] += 1
            
        if gender == 'M':
            gender_cnt[0] += 1
        elif gender == 'F':
            gender_cnt[1] += 1
            
        i += 1
        
    print(age_cnt)
    print(race_cnt)
    print(gender_cnt)