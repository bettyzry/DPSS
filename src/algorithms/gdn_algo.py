import numpy as np
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


class GDNModel:
    def __init__(self, sequence_length=16, stride=5, batch_size=128, num_epochs=100, lr=0.001, dim=64,
                 comment='', seed=0, out_layer_num=1, out_layer_inter_dim=256, decay=0, train_val_pc=0.1, topk=20,
                 device='cuda', save_path=''):

        self.train_config = {'batch': batch_size, 'epoch': num_epochs, 'lr': lr,
                             'slide_win': sequence_length, 'slide_stride': stride,
                             'dim': dim,
                             'comment': comment,
                             'seed': seed,
                             'out_layer_num': out_layer_num,
                             'out_layer_inter_dim': out_layer_inter_dim,
                             'decay': decay,
                             'val_ratio': train_val_pc,
                             'topk': topk,
        }

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic= True

        self.device = device
        self.save_path = save_path
        self.model_save_path = self.get_save_path()[0]

        # env_config = {
        #     'save_path': args.save_path_pattern,
        #     'dataset': args.dataset,
        #     'report': args.report,
        #     'device': args.device,
        #     'load_model_path': args.load_model_path
        # }

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

    def fit(self, train_df):
        # X.interpolate(inplace=True)
        train_df.bfill(inplace=True)

        train_config = self.train_config

        feature_map = list(train_df.columns)
        fc_struc = net_struct.get_fc_graph_struc2(feature_map)
        self.feature_map = feature_map

        fc_edge_index = preprocess.build_loc_net(fc_struc, list(train_df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.fc_edge_index = fc_edge_index

        train_dataset_indata = preprocess.construct_data(train_df, feature_map)
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train',
                                    slide_win=train_config['slide_win'], slide_stride=train_config['slide_stride'])

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])

        self.train_df = train_df
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        edge_index_sets = [fc_edge_index]
        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']).to(self.device)

        train(self.model, self.model_save_path, config=train_config,
              train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader, device=self.device)

    def predict_proba(self, test_df):
        train_config = self.train_config
        self.test_df = test_df

        feature_map = self.feature_map
        fc_edge_index = self.fc_edge_index
        test_dataset_indata = preprocess.construct_data(test_df, feature_map)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test',
                                   slide_win=train_config['slide_win'], slide_stride=train_config['slide_stride'])

        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False)

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
        return test_final_scores_pad

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

    # def evaluate_scores(self, test_final_scores, normal_scores, test_labels):
    #     max_f1, p, r = evaluate.get_max_f1_score(test_final_scores, np.array(test_labels, dtype=int))
    #     print(max_f1, p, r)
    #
    #     print('=========================** Result **============================\n')
    #     info = None
    #     if self.env_config['report'] == 'best':
    #         top1_best_info = evaluate.get_best_performance_data(test_final_scores, test_labels)
    #         info = top1_best_info
    #
    #     # # val_performance: use the maximum anomaly score over the validation dataset to set the threshold=
    #     elif self.env_config['report'] == 'val':
    #         top1_val_info = evaluate.get_val_performance_data(test_final_scores, normal_scores, test_labels)
    #         # top1_best_info = get_best_performance_data(test_scores, test_labels)
    #         info = top1_val_info
    #
    #     print(f'F1 score: {info[0]}')
    #     print(f'precision: {info[1]}')
    #     print(f'recall: {info[2]}\n')

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


if __name__ == '__main__':
    import pandas as pd
    dataname = 'epilepsy'
    datanum = 1
    df = pd.read_csv('data/epilepsy_c/%s_%d.csv' % (dataname, datanum))[['v1', 'v2', 'v3', 'label']]
    size = int(len(df)*0.1)
    train = df[size:, :]
    test = df[:size, :]
    gdn = GDNModel()
    gdn.fit(train)
    result = gdn.predict_proba(test)
    print(result)