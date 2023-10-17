from sklearn.linear_model import Lasso
import numpy as np


class LassoEnsemble:

    def __init__(self, alpha_range=(1e-4, 1e3), num_models=20, ):

        self.best_model = None
        alpha_list = np.geomspace(alpha_range[0], alpha_range[1], num_models)
        self.models = [Lasso()]