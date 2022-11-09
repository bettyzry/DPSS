import autosklearn.classification
import pickle
from src.algorithms.Detector import Detector


class AutoSklearn(Detector):
    def __init__(self, doload=False, time_left_for_this_task=120, per_run_time_limit=30, count=0):
        super(AutoSklearn, self).__init__()
        self.doload = doload
        self.automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                delete_tmp_folder_after_terminate=False,
                resampling_strategy='cv',
                resampling_strategy_arguments={'folds': 5},
            )
        self.count = count
        return

    def fit(self, x, y):
        if self.doload:
            self.automl = self.load('./src/temp/automl_%d.pickle' % self.count)
        else:
            self.automl.fit(x.copy(), y.copy())
            self.automl.refit(x.copy(), y.copy())
            # 在 fit 完毕,选择出最佳模型之后,还需要 refit 一次,这样原来的模型才能根据新的数据进行调整,然后才能进行预测,直接使用 refit 命令无法进行预测,而只是用fit命令,
            # 不适用 refit 命令也会报下面的错错,:
            self.save(self.automl, './src/temp/automl_%d.pickle' % self.count)

    def predict_proba(self, x):
        outlierscore = self.automl.predict_proba(x)
        return outlierscore


