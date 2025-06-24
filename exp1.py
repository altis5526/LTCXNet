import subprocess
import os


if __name__ == "__main__":
  
    # subprocess.run(f'python exp1_train.py --seed {0} --model resnet50', shell=True)

    # train
    for model in ['resnet18', 'resnet50', 'densenet121', 'densenet161', 'convnext', 'vit', 'swin_transformer']:
        if model == 'convnext':
            continue
        # for seed in [0, 123, 61616581]:
        for seed in [0]:
            if os.path.exists(f'C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/exp1/{seed}_{model}'):
                continue
            subprocess.run(f'python exp1_train.py --seed {seed} --model {model}', shell=True)
            
    # test
    for model in ['resnet18', 'resnet50', 'densenet121', 'densenet161', 'convnext', 'vit', 'swin_transformer']:
        if model == 'convnext':
            continue
        # for seed in [0, 123, 61616581]:
        for seed in [0]:
            subprocess.run(f'python exp1_test.py --seed {seed} --model {model}', shell=True)