import torch
import pickle

path = "/mnt/new_usb/jupyter-altis5526/hc3_extra/fairness/results/0228_v9"
seeds = ['0', '123', '222', '33', '4444']
group_ids = [0,1,2,3,4]

for seed in seeds:
    group_sums = []
    with open(f'{path}/{seed}_record.pkl', 'rb') as file:
        record = pickle.load(file)
    for group_id in group_ids:
        sub_test_record = record['test']["ensemble"]['race']
        test_pred, test_target = sub_test_record[group_id]['predict'], sub_test_record[group_id]['target']
        target_sum = torch.sum(test_target, dim=0)
        group_sums.append(target_sum)
    group_sums = torch.stack(group_sums)
    final_sum = torch.sum(group_sums, dim=0)
    print(final_sum)
        
        