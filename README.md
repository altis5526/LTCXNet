## Project File Overview

### 1. Batch Experiment Files
- **Main batch scripts**:  
  - `exp1.py`, `exp2.py`, `run_exp3.py`  
    These scripts handle batch processing for different experiments.  
    You can also understand how to run each experiment individually by inspecting these files.
- **Supporting modules**:  
  - `exp1_*.py`, `exp2_*.py`, `exp3_*.py`  
    These are helper or sub-modules used by the corresponding main scripts.

---

### 2. Utility Scripts
These files perform specific tasks, consistent with their filenames:
- `check_domgraphic.py` – demographic checking  
- `checkGFLOPs.py` – GFLOPs calculation  
- `compute_mf1.py` – Since original code don't support mF1 calculation, this is extra step done after model evaluation

---

### 3. Final Moment Experiment Scripts
These scripts were added at the last minute for extra experiments. Their usefulness is uncertain and they can be ignored if not relevant:
- `final_exp1_test_v2.py`
- `final_test_LTCXNet.py`
- `final_test_others.py`
- `run_final_bootstrap.py` is used to execute the above scripts.

---

### 4. Fairness Evaluation (in `fairness/` folder)
- `run_batch_*.py` – scripts to run fairness experiments
- `v*_test_fairness.py` – scripts for testing fairness
- `v*_check_fairness.py` – scripts for checking fairness results  
  *(Note: Each version serves a different purpose, but I forgot exact purpose.)*

---


