import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from utils.project import project_one_batch, StartEnd_constrain, Priority_constrain
from utils.data_process import TSPDataset

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def compute_loss(assign_matrix, cords_2d):
    # assign_matrix shape: (batch_size, node_num, node_num)
    # cords_2d shape: (batch_size, node_num, node_dim)
    distance_matrix = torch.norm(cords_2d[:, None, :, :] - cords_2d[:, :, None, :], dim=-1)
    
    # Halmination path
    assign_current = assign_matrix[:, :, :-1]
    assign_next = assign_matrix[:, :, 1:]

    edge_prob = torch.matmul(assign_current, assign_next.permute(0, 2, 1))
    expected_distance = (distance_matrix * edge_prob).sum((-1, -2)).mean()

    return expected_distance

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def process_one_batch(
        model,
        batch_input,
        opts,
        constrain_left = None,
        constrain_right = None
):
    
    batch_input = batch_input.to(opts.device)
    

    # get preprocess matrix
    time_now = time.time()
    pre_project_logits = model(batch_input)
    NN_time = time.time() - time_now
    batch_size, node_num, _ = pre_project_logits.shape
    
    # project to the desired space
    time_now = time.time()
    post_project_exp = project_one_batch(pre_project_logits, temp = opts.temp, max_iter = opts.max_iter, \
                                        constrain_left = constrain_left, constrain_right = constrain_right)
    project_time = time.time() - time_now

    # Calculate loss
    cords_2d = batch_input
    loss = compute_loss(post_project_exp, cords_2d)

    return post_project_exp, cords_2d, loss, NN_time, project_time


def train_batch(
        model,
        optimizer,
        epoch,
        batch_id,
        step,
        batch_input,
        opts,
        constrain_left = None,
        constrain_right = None
):
    assign_matrix, cords_2d, loss, _, _ = process_one_batch(model, batch_input, opts, constrain_left, constrain_right)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        grad_norms, grad_norms_clipped = grad_norms
        print('epoch: {}, train_batch_id: {}, avg_cost: {} grad_norm: {}, clipped: {}'.format(\
                        epoch, batch_id, loss.item(), grad_norms[0], grad_norms_clipped[0]))

def validate(model, val_dataset, opts, constrain_left, constrain_right):
    print('Validating...')
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, num_workers=1, drop_last=False, shuffle=False)
    model.eval()
    losses = []
    for batch_id, batch_input in enumerate(tqdm(val_dataloader, disable=opts.no_progress_bar)):
        with torch.no_grad():
            assign_matrix, cords_2d, batch_loss, _, _ = process_one_batch(model, batch_input, opts, constrain_left, constrain_right)
            losses.append(batch_loss.item())
    avg_loss = np.array(losses).mean()

    return avg_loss

    # Put model in eval

def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, opts, log_file):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    log_file.write("Start train epoch {}, lr={} for run {}\n".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = TSPDataset(size=opts.graph_size, num_samples=opts.epoch_size)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1, drop_last=True)

    # Put model in train mode!
    model.train()

    if (opts.task == 'StartEnd'):
        constrain_left, constrain_right = StartEnd_constrain(opts.graph_size)
    elif (opts.task == 'Priority'):
        constrain_left, constrain_right = Priority_constrain(opts.graph_size, opts.priority_level)
    constrain_left = constrain_left.to(opts.device)
    constrain_right = constrain_right.to(opts.device)

    for batch_id, batch_input in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            epoch,
            batch_id,
            step,
            batch_input,
            opts,
            constrain_left,
            constrain_right
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    log_file.write("Finished epoch {}, took {} s\n".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_loss = validate(model, val_dataset, opts, constrain_left, constrain_right)
    print('epoch: {} validation loss: {}'.format(epoch, avg_loss))
    log_file.write('epoch: {} validation loss: {}\n'.format(epoch, avg_loss))

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()