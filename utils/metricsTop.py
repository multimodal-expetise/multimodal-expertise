
from sklearn.metrics import accuracy_score, f1_score
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class MetricsTop():
    def __init__(self, train_mode):
            self.metrics_dict = {
                'Expertise': self.__eval_regression }

    def __eval_regression(self, y_pred, y_true, exclude_zero=False):
        y_pred = np.array([ y.detach().numpy() for y in y_pred])
        y_true =   np.array([ y.detach().numpy() for y in y_true])

        # numeric
        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)

        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = (y_true[non_zeros] > 0)
        binary_preds = (y_pred[non_zeros] > 0)

        # ACC2
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        y_pred = [i[0] for i in y_pred]
        y_true = [i[0] for i in y_true]
        corr = np.corrcoef(y_pred, y_true)[0][1]

        eval_results_reg = {
            "Acc_2":  round(acc2, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results_reg


    def getMetrics(self, datasetName):
        return self.metrics_dict[datasetName]

