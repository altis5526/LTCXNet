import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
import time

def evaluate_macro_f1(y_true, y_pred, thresholds):
    """
    Evaluates the best threshold for maximizing the macro F1-score in a multi-label setting.
    
    Parameters:
    - y_true: Binary ground truth labels for a specific class (shape: n_samples,)
    - y_pred: Probability predictions for a specific class (shape: n_samples,)
    - thresholds: List of thresholds to evaluate
    
    Returns:
    - best_thresh: Optimal threshold for the given class
    - best_f1: Corresponding best macro F1-score
    """
    best_f1 = 0
    best_thresh = 0.5  # Default
    
    for thresh in thresholds:
        f1 = f1_score(y_true, (y_pred >= thresh).astype(int), zero_division=0)
        # print(f"    {thresh} {f1}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    # print('\n')

    return best_thresh, best_f1

def optimize_thresholds(y_true, y_probs):
    """
    Optimizes thresholds for each class in a multi-label classification problem.
    
    Parameters:
    - y_true: True labels (binary, shape: n_samples, n_classes)
    - y_probs: Probability predictions (shape: n_samples, n_classes)
    
    Returns:
    - best_thresholds: List of best thresholds for each class
    """
    num_classes = y_true.shape[1]
    best_thresholds = []

    for class_idx in range(num_classes):
        
        # Get true labels and probabilities for the current class
        y_binary = y_true[:, class_idx]
        class_probs = y_probs[:, class_idx]

        # Search optimal threshold
        thresholds = np.linspace(0.01, 0.99, 100)
        # thresholds = np.linspace(1e-6, 1e-4, 100)     # for bsoftmax
        best_thresh, _ = evaluate_macro_f1(y_binary, class_probs, thresholds)
        best_thresholds.append(best_thresh)

    return np.array(best_thresholds)

def apply_thresholds(y_probs, best_thresholds):
    """
    Applies optimized thresholds to probability predictions in a multi-label setting.
    
    Parameters:
    - y_probs: Probability predictions (shape: n_samples, n_classes)
    - best_thresholds: Optimized thresholds for each class
    
    Returns:
    - y_pred: Final binary predictions (shape: n_samples, n_classes)
    """
    return (y_probs >= best_thresholds).astype(int)

# Example usage
if __name__ == "__main__":

    modes = ['val', 'test']
    model_names = ['LTCXNet', 'LTCXNet_all', 'convnextWithAug',
                   'resnet18', 'resnet50', 'densenet121', 'densenet161', 'convnext', 'vit', 'swin_transformer',
                    'cRT', 'LWS', 'WBCELoss', 'focalLoss', 'LDAMLoss', 'CBLoss', 'bsoftmaxLoss', 'ROS', 'MLROS', 'simCLR']

    data_seeds = [0, 123, 222, 33, 4444]
    
    
    for data_seed in data_seeds:
        for model_name in model_names: 
            
            print(f"Processing {model_name}-{data_seed}.npy")
            time_pin = time.time()
            
            # find best t by val
            result = np.load(f'experiments/tmp/val-{model_name}-{data_seed}.npy', allow_pickle=True).item()
            y_true, y_pred = result["y_true"], result["y_pred"]           
            best_thresholds = optimize_thresholds(y_true, y_pred)
            print(f"    Optimized thresholds: {best_thresholds} | duration {time.time()-time_pin}")
            time_pin = time.time()


            # use best t on test
            result = np.load(f'experiments/tmp/test-{model_name}-{data_seed}.npy', allow_pickle=True).item()
            y_true, y_pred = result["y_true"], result["y_pred"]           

            best_thresholds = best_thresholds
            y_pred = apply_thresholds(y_pred, best_thresholds)
            
            final_macro_f1 = f1_score(y_true, y_pred, average="macro")
            print(f"    Final Macro F1-Score: {final_macro_f1:.4f}")
            
            
            with open(f"experiments/final_test_mF1.txt", 'a') as f:
                f.write(f"{model_name}-{data_seed} macro_f1: {final_macro_f1}\n")
                    
