import subprocess
import os


if __name__ == "__main__":
  
  subprocess.run(f'python exp3_simCLR.py --seed {0}', shell=True)
  
  # subprocess.run(f'python exp3_CBLoss.py --seed {0} --name {"CBLoss_sum"} --loss_type {"sum"}', shell=True)
  # subprocess.run(f'python exp3_CBLoss.py --seed {0} --name {"CBLoss_max"} --loss_type {"max"}', shell=True)  
  # subprocess.run(f'python exp3_LDAM.py --seed {0} --name {"LDAM"}', shell=True)  
  # subprocess.run(f'python exp3_bsoftmax.py --seed {0} --name {"bsoftmax"}', shell=True)
  # subprocess.run(f'python exp3_WBCELoss.py --seed {0} --name {"WBCELoss"}', shell=True)
  subprocess.run(f'python exp3_focalLoss.py --seed {0} --name {"focalLoss"}', shell=True)
  
  # subprocess.run(f'python exp3_cRT.py --seed {0} --name {"cRT"}', shell=True)
  # subprocess.run(f'python exp3_LWS.py --seed {0} --name {"LWS"}', shell=True)
  
  # subprocess.run(f'python exp3_MLROS.py --seed {0} --name {"MLROS"}', shell=True)
  subprocess.run(f'python exp3_MLROS.py --seed {0} --name {"ROS"}', shell=True)