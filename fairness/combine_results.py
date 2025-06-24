import pandas as pd
from statistics import stdev

model_names = ['LTCXNet', 'focalLoss', 'LDAMLoss', 'CBLoss', 'cRT', 'LWS', 'MLROS', 'ROS', 'simCLR', 'WBCELoss', 'bsoftmaxLoss']
seeds = ['0', '123', '222', '33', '4444']
root_dir = "./results"

for model_name in model_names:
    auc_ratios = []
    if model_name == 'LTCXNet':
        for seed in seeds:
            data = pd.read_csv(f"{root_dir}/{model_name}/{seed}_{model_name}_tail.csv")
            auc_ratio = data.iloc[0]['race']
            auc_ratios.append(auc_ratio)
        print(f"ConvNext exp. has auc ratio: {sum(auc_ratios)/len(auc_ratios)} +- {stdev(auc_ratios)}")
        auc_ratios = []
        
        for seed in seeds:
            data = pd.read_csv(f"{root_dir}/{model_name}/{seed}_{model_name}_tail.csv")
            auc_ratio = data.iloc[1]['race']
            auc_ratios.append(auc_ratio)
        print(f"MLD exp. has auc ratio: {sum(auc_ratios)/len(auc_ratios)} +- {stdev(auc_ratios)}")
        auc_ratios = []
        
        for seed in seeds:
            data = pd.read_csv(f"{root_dir}/{model_name}/{seed}_{model_name}_tail.csv")
            auc_ratio = data.iloc[2]['race']
            auc_ratios.append(auc_ratio)
        print(f"Aug exp. has auc ratio: {sum(auc_ratios)/len(auc_ratios)} +- {stdev(auc_ratios)}")
        auc_ratios = []
        
        for seed in seeds:
            data = pd.read_csv(f"{root_dir}/{model_name}/{seed}_{model_name}_tail.csv")
            auc_ratio = data.iloc[3]['race']
            auc_ratios.append(auc_ratio)
        print(f"ensemble exp. has auc ratio: {sum(auc_ratios)/len(auc_ratios)} +- {stdev(auc_ratios)}")
        auc_ratios = []
        
    else:
        for seed in seeds:
            data = pd.read_csv(f"{root_dir}/{model_name}/{seed}_{model_name}_tail.csv")
            auc_ratio = data.iloc[0]['race']
            auc_ratios.append(auc_ratio)

        print(f"{model_name} exp. has auc ratio: {sum(auc_ratios)/len(auc_ratios)} +- {stdev(auc_ratios)}")