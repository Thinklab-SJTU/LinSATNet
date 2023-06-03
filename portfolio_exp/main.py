import os
import torch
from datetime import datetime
from models.handler import train_and_test_loop
import argparse
import pandas as pd
from utils.setting_utils import get_snp500_keys
def main(args):
    before_train = datetime.now().timestamp()
    data_file = os.path.join('dataset', args.dataset + '.csv')
    result_train_file = os.path.join('output', args.dataset, 'train')
    result_test_file = os.path.join('output', args.dataset, 'test')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)
    data = pd.read_csv(data_file)
    snp500 = get_snp500_keys()
    data = data[snp500].pct_change().dropna().values
    data = data[1:-1,1:].astype(float)
    train_ratio = args.train_length / (args.train_length + args.test_length)
    test_ratio = 1 - train_ratio
    train_data = data[:int(train_ratio * len(data))]
    test_data = data[int(train_ratio * len(data))-60:]
    torch.manual_seed(0)
    train_and_test_loop(train_data, test_data, args, result_train_file)
    after_train = datetime.now().timestamp()
    print(f'Training and testing took {(after_train - before_train) / 60} minutes')
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='snp500')
        parser.add_argument('--window_size', type=int, default=120)
        parser.add_argument('--horizon', type=int, default=120)
        parser.add_argument('--train_length', type=float, default=7.5)
        parser.add_argument('--test_length', type=float, default=2.5)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--multi_layer', type=int, default=2)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--norm_method', type=str, default='z')
        parser.add_argument('--optimizer', type=str, default='RMSProp')
        parser.add_argument('--early_stop', type=bool, default=False)
        parser.add_argument('--exponential_decay_step', type=int, default=5)
        parser.add_argument('--decay_rate', type=float, default=0.5)
        parser.add_argument('--dropout_rate', type=float, default=0.95)
        parser.add_argument('--leakyrelu_rate', type=float, default=0.1)
        parser.add_argument('--sharpe_weight', type=float, default=1)
        parser.add_argument('--pred_weight', type=float, default=1)
        parser.add_argument('--use_linsatnet', action='store_true', default=False)
        args = parser.parse_args()
        print(f'Training configs: {args}')
        main(args)
    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    print('Done')