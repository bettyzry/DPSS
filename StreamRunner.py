import numpy as np
import pandas as pd
import configs
import tqdm
import evaluation
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from DPSSOD import DPSSOD
from Framework import Nothing


class StreamRunner():
    def __init__(self, name, winsize, model, params, grid_param, do_refit=False, refit_func='F', refit_size=40, fit_size=10000):
        self.x = None
        self.y = None
        self.init_x = None
        self.init_y = None
        self.name = name
        self.grid_param = grid_param
        self.score = []
        self.label = []
        self.win_size = winsize
        self.model = model
        self.params = params
        self.refit_func = refit_func                # 是否重新训练参数
        self.refit_size = refit_size            # 隔多久重新训练一次
        self.fit_size = fit_size                # 重新训练的时候，用多新的数据
        self.entry_model = model(**params)
        self.eval_result = {'f1': [], 'p': [], 'r': []}
        self.do_refit = do_refit
        if refit_func == 'F' or refit_func == 'automl':
            self.Bayes = Nothing(params)
        elif refit_func == 'skopt':
            self.Bayes = BayesSearchCV(self.model(), self.grid_param, n_iter=30)
        elif refit_func == 'DPSS':
            self.Bayes = DPSSOD(self.model(), self.grid_param, n_init=10, iter=30)
        else:
            print('ERROR fitfunc')

    def setdata(self, init_x, init_y, x, y):
        if isinstance(x, list):
            self.x = np.array(x)
            self.y = np.array(y)
        elif isinstance(x, np.ndarray):
            self.x = x
            self.y = y
        elif isinstance(x, pd.Series):
            self.x = x.values
            self.y = y.values
        else:
            print('This data format (%s) is not supported' % type(x))
            return

        if isinstance(init_x, list):
            self.init_x = np.array(init_x)
            self.init_y = np.array(init_y)
        elif isinstance(init_x, np.ndarray):
            self.init_x = init_x
            self.init_y = init_y
        elif isinstance(init_x, pd.Series):
            self.init_x = init_x.values
            self.init_y = init_y.values
        elif isinstance(init_x, int):
            if init_x == init_y:
                self.init_x = self.x[:init_x]
                self.init_y = self.x[:init_x]
                self.x = self.x[init_x:]
                self.y = self.x[init_x:]
            else:
                print('init_x (%d) != init_y (%d)' % (init_x, init_y))
                return
        elif isinstance(init_x, float) & (init_x < 1) & (init_x > 0):
            if init_x == init_y:
                r = int(init_x * len(x))
                self.init_x = self.x[:r]
                self.init_y = self.y[:r]
                self.x = self.x[r:]
                self.y = self.y[r:]
            else:
                print('init_x (%d) != init_y (%d)' % (init_x, init_y))
                return
        else:
            print('The initial data cannot be set')
            return

    def add(self, x, y):
        """
        This function allows to append data to the already fitted data Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        elif isinstance(x, np.ndarray):
            x = x
            y = y
        elif isinstance(x, pd.Series):
            x = x.values
            y = y.values
        else:
            print('This data format (%s) is not supported' % type(x))
            return
        self.x = np.concatenate((self.x, x), axis=0)
        self.y = np.concatenate((self.y, y), axis=0)
        return

    def initialize(self, init=False):
        if init:
            self.Bayes.fit(self.init_x, self.init_y)
            self.params = self.Bayes.best_params_
            print(self.params)
        self.entry_model = self.model(**self.params)
        self.entry_model.fit(self.init_x, self.init_y)

    def updata_params(self, x, y, count):
        if self.refit_func == 'automl':
            pass
        else:
            self.Bayes.fit(x, y)
            self.params = self.Bayes.best_params_
        print(self.params)

    def refit(self, x, y, count):
        if self.refit_func == 'F':
            pass
        else:
            self.updata_params(x, y, count)
            del self.entry_model
            self.entry_model = self.model(**self.params)
            self.entry_model.fit(x, y)
        return

    def run(self, run_func):
        datasize = len(self.x)
        count = 0
        for i in tqdm.tqdm(range(0, datasize, self.win_size)):
            count += 1
            if self.do_refit and count % self.refit_size == 0:                      # 需要重新训练模型
                self.refit(self.x[max(0, i-self.fit_size): i], self.y[max(0, i-self.fit_size):i], count//self.refit_size)
            if run_func == 0:       # 只跑新输入的数据
                data = self.x[i: i + self.win_size]
                y = self.y[i: i + self.win_size]
                score = self.entry_model.predict_proba(data)[:, 1]
                label = self.entry_model.predict(data, y)
            elif run_func == 1:     # 每次都从头跑
                data = self.x[: i+self.win_size]
                y = self.y[: i+self.win_size]
                score = self.entry_model.predict_proba(data)[:, 1]
                score = score[i:]
                label = self.entry_model.predict(data, y)
                label = label[i:]
            elif run_func == 2:     # 跑前面一小段窗口的数据
                data = self.x[max(0, i-self.fit_size): i+self.win_size]
                y = self.y[max(0, i-self.fit_size): i+self.win_size]
                score = self.entry_model.predict_proba(data)[:, 1]
                score = score[-min(self.win_size, datasize-i):]
                label = self.entry_model.predict(data, y)
                label = label[-min(self.win_size, datasize-i):]
            else:
                score = np.zeros(datasize-i)
                label = np.zeros(datasize-i)
                print('error')
            self.score = np.concatenate([self.score, score])
            self.label = np.concatenate([self.label, label])
            # self.eval(write=True, norm=True, split=True)
            self.eval(write=True, split=False)

    def eval(self, write=True, split=True):
        length = len(self.score)
        if split:       # 只计算当前处理数据的得分
            ypred = self.label[-self.win_size:]
            ytrue = self.y[length-self.win_size:length]
        else:           # 计算平均得分
            ypred = self.label
            ytrue = self.y[:length]

        f1, p, r = evaluation.get_fpr(ytrue, ypred)

        if write:
            self.eval_result['f1'].append(f1)
            self.eval_result['p'].append(p)
            self.eval_result['r'].append(r)
        return f1, p, r

    def plot(self, result):
        datasize = len(self.x)
        x = range(0, datasize, self.win_size)
        fig, ax1 = plt.subplots(figsize=(20, 5))
        for k in result.keys():
            plt.plot(x, result[k], label=k)
        # plt.plot(self.score)
        plt.legend()
        return ax1


def main(func='sesd', dataname='epilepsy', datanum=0, do_refit=False, refit_func='skopt', fit_size=10000, do_init=True):
    from skopt.space import Real, Integer, Categorical
    step = configs.getstep(dataname)
    params = {'sesd': {'seasonality': step, 'weight1': 1, 'weight2': 1, 'weight3': 1},
              'holtwinter': {'weight1': 1, 'weight2': 1, 'weight3': 1, 'alpha': 1, 'beta': 1, 'types': 'linear', 'gama': 1, 'm': step},
              'automl': {'doload': False, 'time_left_for_this_task': 120, 'per_run_time_limit': 30, 'count': 0},
              'knn': {'doload': False, 'n_neighbors': 4, 'algorithm': 'brute'},
              'gdn': {},
              }
    if dataname == 'epilepsy':
        batch_size = Integer(32, 512, name='batch_size', dtype=np.int)
    elif dataname == 'natops':
        batch_size = Integer(32, 256, name='batch_size', dtype=np.int)
    else:
        batch_size = Integer(8, 128, name='batch_size', dtype=np.int)

    grid_param = {
        'gdn': {
            'batch_size': batch_size,
            'lr': Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),}
    }
    model = None
    if func == 'sesd':
        from src.algorithms.sesd import Sesd
        model = StreamRunner(name=func, winsize=step*20, model=Sesd, params=params[func], grid_param=grid_param[func], do_refit=do_refit, refit_func=refit_func, refit_size=5, fit_size=fit_size)
    elif func == 'holtwinter':
        from src.algorithms.holtwinter import HoltWinter
        model = StreamRunner(name=func, winsize=step*20, model=HoltWinter, params=params[func], grid_param=grid_param[func], do_refit=do_refit, refit_func=refit_func, refit_size=5, fit_size=fit_size)
    elif func == 'automl':
        from src.algorithms.autosklearn import AutoSklearn
        model = StreamRunner(name=func, winsize=step*20, model=AutoSklearn, params=params[func], grid_param=grid_param[func], do_refit=do_refit, refit_func=refit_func, refit_size=5, fit_size=fit_size)
    elif func == 'knn':
        from src.algorithms.knn import KNN
        model = StreamRunner(name=func, winsize=step*20, model=KNN, params=params[func], grid_param=grid_param[func], do_refit=do_refit, refit_func=refit_func, refit_size=5, fit_size=fit_size)
    elif func == 'gdn':
        from src.algorithms.Gdn import Gdn
        model = StreamRunner(name=func, winsize=step * 20, model=Gdn, params=params[func], grid_param=grid_param[func], do_refit=do_refit, refit_func=refit_func, refit_size=5, fit_size=fit_size)
    else:
        print('ERROR')
    df = pd.read_csv('data/%s_c/%s_%d.csv' % (dataname, dataname, datanum))
    if dataname == 'epilepsy':
        x = df[['v1', 'v2', 'v3']].values
    elif dataname == 'natops':
        x = df[['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','v7', 'v8', 'v9', 'v10', 'v11', 'v12','v13', 'v14', 'v15', 'v16', 'v17', 'v18','v19', 'v20', 'v21', 'v22', 'v23']].values
    else:
        x = df[['v0', 'v1', 'v2', 'v3', 'v4', 'v5']].values
    y = df['label'].values
    # x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model.setdata(0.1, 0.1, x, y)
    # model.test()
    model.initialize(init=do_init)
    model.run(run_func=0)

    best_f1, best_p, best_r = model.eval(write=False, split=False)
    print('f1: ', best_f1)
    print('p: ', best_p)
    print('r: ', best_r)
    return model


if __name__ == '__main__':
    # test()
    # dataname = 'racket_sports'
    dataname = 'epilepsy'
    main(func='gdn', dataname=dataname, datanum=0, do_refit=False, refit_func='DPSS', fit_size=100*configs.getstep(dataname), do_init=True)
    # init()
    # test()