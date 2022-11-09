from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import numpy as np
import evaluation


class DPSSOD():
    def __init__(self, model, space, n_init=10, iter=30):
        self.optimizer_results = []
        self.best_params = None
        self.best_params_ = None
        self.space = space.values()
        self.model = model
        self.n_init = n_init
        self.iter = iter
        return

    def fit(self, x, y):
        reg = self.model
        space = self.space

        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            reg.fit(x, y)
            return evaluation.scoring_label(reg, x, y)

        res_gp = gp_minimize(objective, space, x0=self.best_params, n_calls=self.iter)
        self.optimizer_results.append(res_gp)

        bestpara = res_gp.x_iters[np.argmax(res_gp.func_vals)]
        if not self.best_params:
            self.best_params = []
        self.best_params.append(bestpara)

        dic_para = {}
        for ii, para in enumerate(res_gp.specs['args']['dimensions'].dimension_names):
            dic_para[para] = bestpara[ii]
        self.best_params_ = dic_para

    def plot(self):
        from skopt.plots import plot_convergence
        plot_convergence(self.best_params[-1])



