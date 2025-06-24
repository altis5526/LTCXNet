import subprocess
import os


if __name__ == "__main__":
  
  # for dataSeed in [0, 123, 222, 33, 4444]:
  #   subprocess.run(f'python v10_test_fairness.py --dataSeed {dataSeed}', shell=True)
    
    
  for dataSeed in [0, 123, 222, 33, 4444]:
    subprocess.run(f'python v10_check_fairnesstype.py --dataSeed {dataSeed}', shell=True)