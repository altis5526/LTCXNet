import torch
import pickle

def compare_state_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1["model_state_dict"]:
        print(dict2["model_state_dict"][key])
        break
        if not torch.equal(dict1["model_state_dict"][key], dict2["model_state_dict"][key]):
            return False
    return True

def compare_pkl(dict1, dict2):
    
    return True


# folder = "/mnt/new_usb/jupyter-altis5526/hc3_extra"

# weights1 = torch.load(f"{folder}/weights/LDAMLoss/model_best.pt", map_location='cpu')
# weights2 = torch.load(f"{folder}/weights/CBLoss/model_best.pt", map_location='cpu')
# are_equal = compare_state_dicts(weights1, weights2)
# print("Are the weights identical?", are_equal)

model_name = ["focalLoss", "cRT"]

path = f"/mnt/new_usb/jupyter-altis5526/hc3_extra/fairness/results/0609_v10"
with open(f'{path}/0_{model_name[0]}_test.pkl', 'rb') as file:
    record0 = pickle.load(file)

with open(f'{path}/0_{model_name[1]}_test.pkl', 'rb') as file:
    record1 = pickle.load(file)
