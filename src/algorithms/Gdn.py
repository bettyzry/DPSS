import numpy as np
import pandas as pd
import torch
import os
import random
import string
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from pathlib import Path

from src.algorithms.gdn.models import GDN
from src.algorithms.gdn.datasets.TimeDataset import TimeDataset
from src.algorithms.gdn.util import data, preprocess, net_struct
from src.algorithms.gdn.run import train, test
import configs
from src.algorithms.Detector import Detector


class Gdn(Detector):
    def __init__(self, sequence_length=16, stride=5, batch_size=128, num_epochs=100, lr=0.001, dim=64,
                     comment='', seed=0, out_layer_num=1, out_layer_inter_dim=256, decay=0, train_val_pc=0.1, topk=3,
                     device='cuda', save_path=''):
        super(Gdn, self).__init__()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.sequence_length = sequence_length
        self.stride = stride
        self.dim = dim
        self.comment = comment
        self.seed = seed
        self.out_layer_num = out_layer_num
        self.out_layer_inter_dim = out_layer_inter_dim
        self.decay = decay
        self.train_val_pc = train_val_pc
        self.topk = topk

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = device
        self.save_path = save_path
        self.model_save_path = self.get_save_path()[0]

        self.train_df = None
        self.test_df = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.feature_map = None
        self.fc_edge_index = None

        self.model = None
        return

    def fit(self, x, y):
        datasize = x.shape[1]
        column = ['A%d' % i for i in range(1, datasize+1)]
        train_df = pd.DataFrame(x, columns=column)

        # X.interpolate(inplace=True)
        train_df.bfill(inplace=True)

        feature_map = list(train_df.columns)
        fc_struc = net_struct.get_fc_graph_struc2(feature_map)
        self.feature_map = feature_map

        fc_edge_index = preprocess.build_loc_net(fc_struc, list(train_df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.fc_edge_index = fc_edge_index

        train_dataset_indata = preprocess.construct_data(train_df, feature_map)
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train',
                                    slide_win=self.sequence_length, slide_stride=self.stride)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, self.seed, self.batch_size,
                                                            val_ratio=self.train_val_pc)

        self.train_df = train_df
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        edge_index_sets = [fc_edge_index]
        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=self.dim,
                         input_dim=self.sequence_length,
                         out_layer_num=self.out_layer_num,
                         out_layer_inter_dim=self.out_layer_inter_dim,
                         topk=self.topk).to(self.device)

        train(self.model, self.model_save_path, lr=self.lr, decay=self.decay, epoch=self.num_epochs,
              train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader, device=self.device)

    def predict_proba(self, x):
        datasize = x.shape[1]
        column = ['A%d' % i for i in range(1, datasize+1)]
        test_df = pd.DataFrame(x, columns=column)

        self.test_df = test_df

        feature_map = self.feature_map
        fc_edge_index = self.fc_edge_index
        test_dataset_indata = preprocess.construct_data(test_df, feature_map)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test',
                                   slide_win=self.sequence_length, slide_stride=self.stride)

        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # test
        self.model.load_state_dict(torch.load(self.model_save_path))
        best_model = self.model.to(self.device)

        # test_result is [test_predicted_list, test_ground_list, test_labels_list]
        _, test_result = test(best_model, self.test_dataloader, device=self.device)
        _, val_result = test(best_model, self.val_dataloader, device=self.device)

        test_scores = self.get_full_err_scores(test_result)
        normal_scores = self.get_full_err_scores(val_result)
        test_final_scores = self.get_final_err_scores(test_scores, topk=1)

        # padding scores of the beginning part of time-series
        padding_list = np.zeros(test_df.shape[0] - len(test_final_scores))
        test_final_scores_pad = np.hstack([padding_list, test_final_scores])

        # padding_list = np.zeros([test_scores.shape[0], test_df.shape[0] - test_scores.shape[1]])
        # transfer to shape [n_samples, n_dim]
        # test_scores_pad = np.concatenate([padding_list, test_scores], axis=1).T

        # predictions_dic = {'score_t': test_final_scores_pad,
        #                    'score_tc': test_scores_pad,
        #                    'error_t': None,
        #                    'error_tc': None,
        #                    'recons_tc': None,
        #                    }
        outlierscore = np.array([[1-i, i] for i in test_final_scores_pad])
        return outlierscore

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def get_full_err_scores(self, test_result):
        np_test_result = np.array(test_result)
        feature_num = np_test_result.shape[-1]
        n_sample = np_test_result.shape[1]

        all_scores = np.zeros([feature_num, n_sample])
        for i in range(feature_num):
            # predict & real-value
            test_predict, test_gt = np_test_result[:2, :, i]
            scores = self.get_err_scores(test_predict, test_gt)
            all_scores[i] = scores
        return all_scores

    def get_err_scores(self, test_predict, test_gt):
        n_err_mid, n_err_iqr = data.get_err_median_and_iqr(test_predict, test_gt)
        test_delta = np.abs(np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_gt).astype(np.float64)))
        epsilon = 1e-2
        err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3
        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])
        return smoothed_err_scores

    def get_final_err_scores(self, total_err_scores, topk=1):
        total_features = total_err_scores.shape[0]
        # 将第k小的数字，放在从前往后的第k个位置，range可以指定多个位置
        # 权衡多个维度上的分数结果？获得每个维度上top_k分数的索引
        topk_indices = np.argpartition(total_err_scores,
                                       range(total_features - topk - 1, total_features), axis=0)[-topk:]
        # 将每个时间点，分数最高的top-k个维度上的分数相加
        total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
        return total_topk_err_scores

    def get_save_path(self):
        # dir_path = self.env_config['save_path']
        dir_path = self.save_path

        now = datetime.now()
        datestr = now.strftime('%m-%d-%H.%M.%S')
        mask = ''.join(random.sample(string.ascii_letters, 8))

        paths = [
            os.path.join(configs.intermediate_dir, 'gdn_saved_model', f'best_{datestr}{mask}.pt'),
            os.path.join(configs.intermediate_dir, 'gdn_saved_model', f'{dir_path}/{datestr}{mask}.pt')
            # f'{configs}z-intermediate_model_files./pretrained/{dir_path}/best_{datestr}{mask}.pt',
            # f'./results/{dir_path}/{datestr}{mask}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths