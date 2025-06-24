import subprocess
import os


if __name__ == "__main__":
  
    # subprocess.run(f'python train_submodel.py --seed {0}', shell=True)

    # train final model
    # for seed in [0, 123, 61616581]:
    #     if os.path.exists(f'C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/exp2/{seed}_final_model'):
    #         continue
    #     subprocess.run(f'python exp2_train.py --seed {seed}', shell=True)
        
    # test final model
    # for seed in [0, 123, 61616581]:
    #     subprocess.run(f'python exp2_test.py --seed {seed}', shell=True)
        
    
    # subprocess.run(f'python exp2_train_ablation.py --seed {0} --id {1}', shell=True)
    # train ablation model
    for seed in [0, 123, 61616581]:
        for id in [1, 2, 3, 4]:
            if os.path.exists(f'C:/Users/112062522/Downloads/112062522_whuang/research/hc2/experiments/exp2/{seed}_model_{id}'):
                continue
            subprocess.run(f'python exp2_train_ablation.py --seed {seed} --id {id}', shell=True)
            
            