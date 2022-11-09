import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import metrics
import utils
import configs


def get_fpr(ytrue, ypred):
    p = precision_score(ytrue, ypred)
    r = recall_score(ytrue, ypred)
    f1 = 2 * p * r / (p + r + 1e-5)
    return f1, p, r


def get_best_label(ytrue, score):
    # print(sum(np.isnan(score)))
    # print(score)
    precision, recall, threshold = metrics.precision_recall_curve(y_true=ytrue, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    threshold = threshold[np.argmax(f1)]
    ypred = utils.score2label_thresholdvalue(score, threshold)    # 让f1最大的threshold的位置
    return ypred


def scoring_score(estimator, x, y):
    outlierscore = estimator.predict_proba(x)[:, 1]
    outlierscore = adjust_scores(y, outlierscore)
    best_f1, _, _ = get_best_f1(y, outlierscore)
    return best_f1


def scoring_label(estimator, x, y):
    label = estimator.predict(x, y)
    f, p, r = get_fpr(y, label)
    return f


def get_preformance(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr


def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def evaluation(score, ytrue):
    auroc = metrics.roc_auc_score(ytrue, score)
    ap = metrics.average_precision_score(y_true=ytrue, y_score=score, average=None)
    best_f1, best_p, best_r = get_best_f1(ytrue, score)
    return auroc, ap, best_f1, best_p, best_r


def norm_evaluation2(score, ytrue, threshold):
    event_num = int(len(ytrue)//configs.step)

    ypred = utils.score2label_threshold(score, threshold)
    n_score = adjust_scores(ytrue, score)
    n_ytrue = [ytrue[i*configs.step] for i in range(event_num)]
    n_ypred = [sum(ypred[i*configs.step: (i+1)*configs.step]) for i in range(event_num)]
    n_ypred = [1 if i > 0 else 0 for i in n_ypred]

    auroc = metrics.roc_auc_score(ytrue, n_score)
    ap = metrics.average_precision_score(y_true=ytrue, y_score=n_score, average=None)
    p = precision_score(n_ytrue, n_ypred)
    r = recall_score(n_ytrue, n_ypred)
    fscore = f1_score(n_ytrue, n_ypred)

    return auroc, ap, p, r, fscore


def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score
