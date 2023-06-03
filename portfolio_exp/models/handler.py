import json
from datetime import datetime
from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os
from matplotlib import pyplot as plt

from utils.math_utils import evaluate, compute_measures, compute_sharpe_ratio
from LinSATNet import linsat_layer

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def train_and_test_loop(train_data, valid_data, args, result_file):
    node_cnt = train_data.shape[1]
    model = Model(units=node_cnt, stack_cnt=2, time_step=args.window_size, multi_layer=args.multi_layer, horizon=args.horizon, dropout_rate=args.dropout_rate, leaky_rate=args.leakyrelu_rate)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_sharpe_ratio = np.inf
    validate_score_non_decrease_count = 0
    to_plot=[]
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()
            forecast, _,weight = model(inputs)
            loss = forecast_loss(forecast, target)
            pred_weight = args.pred_weight
            sharpe_weight = args.sharpe_weight
            loss = loss * pred_weight
            for num in range(target.shape[0]):
                mu, cov = compute_measures(target[num])
                sharpe = compute_sharpe_ratio(mu, cov, weight[num], 0.03)
                loss += - sharpe_weight * sharpe
                mu, cov = compute_measures(target[num])
                c = torch.zeros((1,493))
                c[0,0:5] = 1
                d = torch.tensor([0.5])
                d = d.float()
                f = torch.tensor([1.0])
                e = torch.zeros((1,493))
                e[0,:] = 1
                e = e.to(args.device)
                f = f.to(args.device)
                c = c.to(args.device)
                d = d.to(args.device)
                if args.use_linsatnet ==True:
                    probs = linsat_layer(weight[num],E=e,f=f,C=c,d=d,tau=0.01)
                else:
                    probs = weight[num]
                sharpe = compute_sharpe_ratio(mu, cov, probs, 0.03)
                loss += -sharpe_weight * sharpe            
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        
        model.eval()
        sharpe_test = 0
        count = 0
        for j, (inputs, target) in enumerate(valid_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            forecast, _,weight = model(inputs)
            for num in range(target.shape[0]):
                count += 1
                mu, cov = compute_measures(target[num])
                mu_pre, cov_pre = compute_measures(forecast[num])
                c = torch.zeros((1,493))
                c[0,0:5] = 1
                d = torch.tensor([0.5])
                d = d.float()
                f = torch.tensor([1.0])
                e = torch.zeros((1,493))
                e[0,:] = 1
                c = c.to(args.device)
                d = d.to(args.device)
                e = e.to(args.device)
                f = f.to(args.device)
                if args.use_linsatnet ==True:
                    probs = linsat_layer(weight[num],E=e,f=f,C=c,d=d,tau=0.01)
                else:
                    probs = weight[num]
                sharpe = compute_sharpe_ratio(mu, cov, probs, 0.03)
                sharpe_test += sharpe
        average_sharpe_test = sharpe_test/count
        print("The average sharpe ratio is:", average_sharpe_test)
        
        
        to_plot.append((sharpe_test/count).detach().cpu()) 
        save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if average_sharpe_test > best_validate_sharpe_ratio:
            save_model(model, result_file)
            validate_score_non_decrease_count = 0
        else:
            validate_score_non_decrease_count += 1
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
            
    plt.plot(to_plot)
    plt.savefig("sharpe_ratio")


