import torch
import numpy as np
from LinSATNet import linsat_layer

def StartEnd_constrain(node_num):
    #transfer the constrain Sum_i X_ij = 1, Sum_j Xij = 1, X_s1 = 1, X_en = 1, to Ax = b, x is the flattened X
    #this function return the required A and b
    b = torch.ones(2 * node_num + 2, dtype=torch.float32)
    A = torch.zeros([2 * node_num + 2, node_num * node_num], dtype=torch.float32)
    
    #the row constrain
    for i in range(node_num):
        A[i, node_num * i : node_num * (i + 1)] = 1

    #the column constrain
    column_gap = node_num * torch.arange(node_num)
    for i in range(node_num):
        A[node_num + i, i + column_gap] = 1

    #start and end node constrain, set to 0.999 to avoid numerical issue
    A[2 * node_num, 0] = 1; b[2 * node_num] = 0.999
    A[2 * node_num + 1, -1] = 1; b[2 * node_num + 1] = 0.999
    
    return A, b

def Priority_constrain(node_num, priority_level = 6):
    #Start-End constrain with priority customer, the second node should be visited within priority_level steps

    b = torch.ones(2 * node_num + 3, dtype=torch.float32)
    A = torch.zeros([2 * node_num + 3, node_num * node_num], dtype=torch.float32)
    
    #the row constrain
    for i in range(node_num):
        A[i, node_num * i : node_num * (i + 1)] = 1

    #the column constrain
    column_gap = node_num * torch.arange(node_num)
    for i in range(node_num):
        A[node_num + i, i + column_gap] = 1

    #start and end node constrain
    A[2 * node_num, 0] = 1; b[2 * node_num] = 0.999
    A[2 * node_num + 1, -1] = 1; b[2 * node_num + 1] = 0.999

    #second node should be visited within priority_level steps
    A[2 * node_num + 2, node_num : node_num + priority_level + 1] = 1; b[2 * node_num + 2] = 0.999
    
    return A, b


def project_one_batch(pre_project_logits, temp = 1e0, max_iter = 100, constrain_left = None, constrain_right = None):
    batch_size, node_num, _ = pre_project_logits.shape
    pre_project_logits_flatten = pre_project_logits.reshape(-1, node_num * node_num)
    post_project_exp_flatten = linsat_layer(x = pre_project_logits_flatten, E = constrain_left, f = constrain_right, tau = temp, max_iter = max_iter, no_warning=True)
    post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    
    return post_project_exp