import subprocess
import os


if __name__ == "__main__":  

  # ablation
  for dataSeed in [0, 123, 222, 33, 4444]:
    subprocess.run(f'python final_test_LTCXNet.py --dataSeed {dataSeed} --name {"LTCXNet"}', shell=True)
    subprocess.run(f'python final_test_LTCXNet.py --dataSeed {dataSeed} --name {"LTCXNet_all"}', shell=True)
    subprocess.run(f'python final_exp1_test_v2.py --dataSeed {dataSeed} --model {"convnextWithAug"}', shell=True)
   
  
  # bakbone
  for dataSeed in [0, 123, 222, 33, 4444]:
    for model in ['resnet18', 'resnet50', 'densenet121', 'densenet161', 'convnext', 'vit', 'swin_transformer']:       
      subprocess.run(f'python final_exp1_test_v2.py --dataSeed {dataSeed} --model {model}', shell=True)  
  
  
  
  # comparison with others
  for dataSeed in [0, 123, 222, 33, 4444]:   
    for model in ['cRT', 'LWS', 'WBCELoss', 'focalLoss', 'LDAMLoss', 'CBLoss', 'bsoftmaxLoss', 'ROS', 'MLROS', 'simCLR']:  
      
      # if model == 'bsoftmaxLoss':
      #   continue
      
      subprocess.run(f'python final_test_others.py --dataSeed {dataSeed} --model {model}', shell=True)
     
      
  # subprocess.run(f'python final_test_others.py --dataSeed {0} --model {"bsoftmaxLoss"}', shell=True)