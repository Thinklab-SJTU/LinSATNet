# Please refer to the following ICML paper for details:
# Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang and Junchi Yan.
# LinSATNet: The Positive Linear Satisfiability Neural Networks. ICML 2023.
#
# Code author: Runzhong Wang (runzhong.wang@outlook.com)

import torch
import sys


def linsat_layer(x, A=None, b=None, C=None, d=None, E=None, f=None, constr_dict=None, tau=0.05, max_iter=100, dummy_val=0,
                 mode='v2', grouped=True, no_warning=False):
    """
    Project x with the constraints A x <= b, C x >= d, E x = f.
    All elements in A, b, C, d, E, f must be non-negative.

    :param x: (n_v), it can optionally have a batch size (b x n_v)
    :param A, C, E: (n_c x n_v)
    :param b, d, f: (n_c)
    :param constr_dict: a dictionary with initialized constraint information, which is the output of the function
        `init_constraints`. Specifying this variable could avoid re-initializing the constraints for the same
        constraints and improve the efficiency
    :param tau: parameter to control hard/soft constraint
    :param max_iter: max number of iterations
    :param dummy_val: value of dummy variable
    :param grouped: group non-overlapping constraints in one operation for better efficiency
    :param mode: v1 or v2
    :param no_warning: turn off warning message
    :return: (n_v) or (b x n_v), the projected variables
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        vector_input = True
    elif len(x.shape) == 2:
        vector_input = False
    else:
        raise ValueError('input data shape not understood.')

    batch_size = x.shape[0]
    num_var = x.shape[1]

    if constr_dict is None:
        constr_dict = init_constraints(num_var, A, b, C, d, E, f, grouped)

    param_dict = {'tau': tau, 'max_iter': max_iter, 'dummy_val': dummy_val, 'batch_size': batch_size,
                  'num_var': num_var, 'no_warning': no_warning}
    param_dict.update(constr_dict)

    is_sparse = param_dict.pop('is_sparse')

    if grouped and is_sparse and mode == 'v2':
        kernel = linsat_kernel_grouped_sparse_v2
    elif grouped and not is_sparse and mode == 'v2':
        kernel = linsat_kernel_grouped_dense_v2
    elif not grouped and not is_sparse and mode == 'v1':
        kernel = linsat_kernel_v1
    elif not grouped and not is_sparse and mode == 'v2':
        kernel = linsat_kernel_v2
    else:
        raise NotImplementedError(f'mode={mode}, grouped={grouped}, is_sparse={is_sparse}')

    x = kernel(x, **param_dict)

    if vector_input:
        x = x.squeeze(0)
    return x


def init_constraints(num_var, A=None, b=None, C=None, d=None, E=None, f=None, grouped=True):
    """
    Initialize the constraint into a state dict (can be reused to improve efficiency)

    :param num_var: number of variables
    """
    num_constr = 0

    device = None
    is_sparse = None
    for _ in (A, C, E):
        if _ is not None:
            device = _.device
            is_sparse = _.is_sparse
            break
    if device is None: raise ValueError(f'At least one variable of A, C, E should be specified')

    A, b, num_constr = _init_shape(A, b, num_var, num_constr, device, is_sparse)
    C, d, num_constr = _init_shape(C, d, num_var, num_constr, device, is_sparse)
    E, f, num_constr = _init_shape(E, f, num_var, num_constr, device, is_sparse)
    ori_A, ori_b, ori_C, ori_d, ori_E, ori_f = A, b, C, d, E, f

    if grouped:
        if is_sparse:
            dummy_A = torch.sparse_coo_tensor(torch.arange(ori_b.shape[0], device=ori_A.device)[None, :].repeat(2, 1), ori_b,
                                              [ori_b.shape[0], ori_b.shape[0]])
        else:
            dummy_A = torch.diagflat(ori_b)
        A = torch.cat((
            ori_A,
            dummy_A,
            torch.zeros(ori_A.shape[0], ori_C.shape[0] + ori_E.shape[0], layout=ori_A.layout, device=ori_A.device),
        ), dim=-1)  # n_c_a x (n_v + n_c), each constraint has a dummy var
        b = torch.stack((ori_b, _sum_dim(ori_A, dim=-1)), dim=-1)  # n_c x 2

        gamma = torch.floor(_sum_dim(ori_C, dim=-1) / ori_d)
        if is_sparse:
            dummy_C = torch.sparse_coo_tensor(torch.arange(ori_d.shape[0], device=ori_C.device)[None, :].repeat(2, 1), gamma * ori_d,
                                              [ori_d.shape[0], ori_d.shape[0]])
        else:
            dummy_C = torch.diagflat(gamma * ori_d)
        C = torch.cat((
            ori_C,
            torch.zeros(ori_C.shape[0], ori_A.shape[0], layout=ori_C.layout, device=ori_C.device),
            dummy_C,
            torch.zeros(ori_C.shape[0], ori_E.shape[0], layout=ori_C.layout, device=ori_C.device),
        ), dim=-1)  # n_c_c x (n_v + n_c)
        d = torch.stack(((gamma + 1) * ori_d, _sum_dim(ori_C, dim=-1) - ori_d), dim=-1)  # n_c x 2

        E = torch.cat((
            ori_E,
            torch.zeros(ori_E.shape[0], num_constr, layout=ori_E.layout, device=ori_E.device)
        ), dim=-1)  # n_c_e x (n_v + n_c)
        f = torch.stack((ori_f, _sum_dim(ori_E, dim=-1) - ori_f), dim=-1)  # n_c x 2
    else:
        A = torch.cat((A, b.unsqueeze(-1)), dim=-1) # n_c x n_v
        b = torch.stack((b, A[:, :-1].sum(dim=-1)), dim=-1) # n_c x 2

        gamma = torch.floor(C.sum(dim=-1) / d)
        C = torch.cat((C, (gamma * d).unsqueeze(-1)), dim=-1) # n_c x n_v
        d = torch.stack(((gamma + 1) * d, C[:, :-1].sum(dim=-1) - d), dim=-1) # n_c x 2

        E = torch.cat((E, torch.zeros_like(f).unsqueeze(-1)), dim=-1) # n_c x n_v
        f = torch.stack((f, E[:, :-1].sum(dim=-1) - f), dim=-1) # n_c x 2

    # merge constraints
    A = torch.cat((A, C, E), dim=0) # n_c x n_v
    b = torch.cat((b, d, f), dim=0) # n_c x 2
    if is_sparse: A = A.coalesce()

    if torch.any(b == 0):
        b += 1e-7  # handle numerical issue

    # normalize values
    if not torch.all(torch.abs(_sum_dim(A, dim=-1) - b.sum(dim=-1)) < 1e-4):
        raise RuntimeError('Marginal distributions are not matched! Please report this issue on GitHub.')
    A = _elem_div(A, _sum_dim(A, dim=-1), dim=-1)
    b = b / b.sum(dim=-1, keepdim=True)

    ori_dict = {'ori_A': ori_A, 'ori_b': ori_b, 'ori_C': ori_C, 'ori_d':ori_d, 'ori_E': ori_E, 'ori_f': ori_f}

    if not grouped:
        constr_dict = {'A': A, 'b': b}
    else:
        # group non-overlapping constraints for better efficiency
        constr_dict = _group_constr_greedy(A, b, num_var, num_constr)
    constr_dict['num_constr'] = num_constr
    constr_dict['is_sparse'] = is_sparse
    constr_dict.update(ori_dict)
    return constr_dict


################################################################################
#                           LinSATNet Kernel functions                         #
################################################################################

def linsat_kernel_v1(x, A, b, tau, max_iter, dummy_val,
                     batch_size, num_var, num_constr, ori_A, ori_b, ori_C, ori_d, ori_E, ori_f,
                     no_warning):
    # add dummy variables
    dum_x1 = []
    dum_x2 = []
    for j in range(num_constr):
        dum_x1.append(torch.full((batch_size, 1), dummy_val, dtype=x.dtype, device=x.device))
        dum_x2.append(torch.full((batch_size, torch.sum(A[j] != 0)), dummy_val, dtype=x.dtype, device=x.device))

    # operations are performed on log scale
    log_x = x / tau
    log_dum_x1 = [d / tau for d in dum_x1]
    log_dum_x2 = [d / tau for d in dum_x2]
    last_log_x = log_x

    log_A = torch.log(A)
    log_b = torch.log(b)

    if torch.any(torch.isinf(log_b)): raise RuntimeError('Inf encountered in log_b!')
    if torch.any(torch.isnan(log_A)): raise RuntimeError('Nan encountered in log_A!')
    if torch.any(torch.isnan(log_b)): raise RuntimeError('Nan encountered in log_b!')

    # Multi-set marginal Sinkhorn iterations
    for i in range(max_iter):
        for j in range(num_constr):
            _log_x = torch.cat((log_x, log_dum_x1[j]), dim=-1) # batch x n_v

            nonzero_indices = torch.where(A[j] != 0)[0]

            log_nz_x = _log_x[:, nonzero_indices]
            log_nz_x = torch.stack((log_nz_x, log_dum_x2[j]), dim=1)  # batch x 2 x n_v

            log_nz_Aj = log_A[j][nonzero_indices].unsqueeze(0).unsqueeze(0)
            log_t = log_nz_x + log_nz_Aj

            log_sum = torch.logsumexp(log_t, 2, keepdim=True) # batch x 2 x 1
            log_t = log_t - log_sum + log_b[j].unsqueeze(0).unsqueeze(-1)

            log_sum = torch.logsumexp(log_t, 1, keepdim=True) # batch x 1 x n_v
            log_t = log_t - log_sum + log_nz_Aj
            log_nz_x = log_t - log_nz_Aj

            log_dum_x1[j] = log_nz_x[:, 0, -1:]
            log_dum_x2[j] = log_nz_x[:, 1, :]
            if A[j][-1] != 0:
                scatter_idx = nonzero_indices[:-1].unsqueeze(0).repeat(batch_size, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, 0, :-1])
            else:
                scatter_idx = nonzero_indices.unsqueeze(0).repeat(batch_size, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, 0, :])

        diff = torch.max(torch.norm((log_x - last_log_x).view(batch_size, -1), dim=-1))
        cv_Ab = torch.matmul(ori_A, torch.exp(log_x).t()).t() - ori_b.unsqueeze(0)
        cv_Cd = -torch.matmul(ori_C, torch.exp(log_x).t()).t() + ori_d.unsqueeze(0)
        cv_Ef = torch.abs(torch.matmul(ori_E, torch.exp(log_x).t()).t() - ori_f.unsqueeze(0))

        if diff <= 1e-3 and \
                torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) < 1e-3:
            break
        last_log_x = log_x

    if not no_warning and \
            torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) > 0.1 * batch_size:
        print('Warning: non-zero constraint violation within max iterations. Add more iterations or '
              'the problem may be infeasible',
              file=sys.stderr)

    return torch.exp(log_x)


def linsat_kernel_v2(x, A, b, tau, max_iter, dummy_val,
                     batch_size, num_var, num_constr, ori_A, ori_b, ori_C, ori_d, ori_E, ori_f,
                     no_warning):
    # add dummy variables
    dum_x1 = []
    for j in range(num_constr):
        dum_x1.append(torch.full((batch_size, 2, 1), dummy_val, dtype=x.dtype, device=x.device))
    dum_x2 = torch.full((batch_size, num_var), dummy_val, dtype=x.dtype, device=x.device)

    # operations are performed on log scale
    log_x = x / tau
    log_dum_x1 = [d / tau for d in dum_x1]
    log_dum_x2 = dum_x2 / tau

    # perform a row norm first
    log_x = torch.stack((log_x, log_dum_x2), dim=1) # batch x 2 x n_v
    log_sum = torch.logsumexp(log_x, 1, keepdim=True)  # batch x 1 x (n_v+1)
    log_x = log_x - log_sum

    log_A = torch.log(A)
    log_b = torch.log(b)

    if torch.any(torch.isinf(log_b)): raise RuntimeError('Inf encountered in log_b!')
    if torch.any(torch.isnan(log_A)): raise RuntimeError('Nan encountered in log_A!')
    if torch.any(torch.isnan(log_b)): raise RuntimeError('Nan encountered in log_b!')

    # Multi-set marginal Sinkhorn iterations
    for i in range(max_iter):
        num_sat_constrs = 0
        for j in range(num_constr):
            _log_x = torch.cat((log_x, log_dum_x1[j]), dim=-1) # batch x 2 x (n_v+1)

            nonzero_indices = torch.where(A[j] != 0)[0]

            log_nz_x = _log_x[:, :, nonzero_indices]

            log_nz_Aj = log_A[j][nonzero_indices].unsqueeze(0).unsqueeze(0)

            log_sum = torch.logsumexp(log_nz_x + log_nz_Aj, 2, keepdim=True) # batch x 2 x 1
            if torch.all(torch.abs(log_sum - log_b[j].unsqueeze(0).unsqueeze(-1)) < 1e-3):
                num_sat_constrs += 1
                continue
            log_nz_x = log_nz_x - log_sum + log_b[j].unsqueeze(0).unsqueeze(-1)

            log_sum = torch.logsumexp(log_nz_x, 1, keepdim=True) # batch x 1 x (n_v+1)
            log_nz_x = log_nz_x - log_sum

            if A[j][-1] != 0:
                scatter_idx = nonzero_indices[:-1].unsqueeze(0).unsqueeze(1).repeat(batch_size, 2, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, :, :-1])
                log_dum_x1[j] = log_nz_x[:, :, -1:]
            else:
                scatter_idx = nonzero_indices.unsqueeze(0).unsqueeze(1).repeat(batch_size, 2, 1)
                log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, :, :])

        if num_sat_constrs == num_constr:
            break

    x = torch.exp(log_x[:, 0, :]) # remove dummy row & transform from log scale

    with torch.no_grad():
        cv_Ab = torch.matmul(ori_A, x.t()).t() - ori_b.unsqueeze(0)
        cv_Cd = -torch.matmul(ori_C, x.t()).t() + ori_d.unsqueeze(0)
        cv_Ef = torch.abs(torch.matmul(ori_E, x.t()).t() - ori_f.unsqueeze(0))
        if not no_warning and \
                torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) > 0.1 * batch_size:
            print('Warning: non-zero constraint violation within max iterations. Add more iterations or '
                  'the problem may be infeasible',
                  file=sys.stderr)

    return x


def linsat_kernel_grouped_dense_v2(x, grouped_A, grouped_b, grouped_nz, tau, max_iter, dummy_val,
                                   batch_size, num_var, num_constr, ori_A, ori_b, ori_C, ori_d, ori_E, ori_f,
                                   no_warning):
    # add dummy variables
    dum_x1 = torch.full((batch_size, 2, num_constr), dummy_val, dtype=x.dtype, device=x.device)
    dum_x2 = torch.full((batch_size, num_var), dummy_val, dtype=x.dtype, device=x.device)

    # operations are performed on log scale
    log_x = x / tau
    log_dum_x1 = dum_x1 / tau
    log_dum_x2 = dum_x2 / tau

    log_gA = [torch.log(A) for A in grouped_A]
    log_gb = [torch.log(b) for b in grouped_b]

    # perform a row norm first
    log_x = torch.stack((log_x, log_dum_x2), dim=1) # batch x 2 x n_v
    log_x = torch.cat((log_x, log_dum_x1), dim=2) # batch x 2 x (n_v + n_c)
    log_sum = torch.logsumexp(log_x, 1, keepdim=True)  # batch x 1 x (n_v + n_c)
    log_x = log_x - log_sum

    if any([torch.any(torch.isinf(log_b)) for log_b in log_gb]): raise RuntimeError('Inf encountered in log_b!')
    if any([torch.any(torch.isnan(log_A)) for log_A in log_gA]): raise RuntimeError('Nan encountered in log_A!')
    if any([torch.any(torch.isnan(log_b)) for log_b in log_gb]): raise RuntimeError('Nan encountered in log_b!')

    # Multi-set marginal Sinkhorn iterations
    for i in range(max_iter):
        num_sat_constrs = 0
        for log_nz_Aj, log_bj, nz_indices in zip(log_gA, log_gb, grouped_nz):
            num_cur_constr = log_nz_Aj.shape[1]
            log_nz_Aj = log_nz_Aj[None, :, None, :] # 1 x n_c x 1 x n_v
            log_bj = log_bj[None, :, :, None] # 1 x n_c x 2 x 1
            log_nz_x = log_x[:, :, nz_indices] # batch x 2 x n_v

            log_sum = torch.logsumexp(log_nz_x[:, None, :, :] + log_nz_Aj, 3, keepdim=True) # batch x n_c x 2 x 1
            if torch.all(torch.abs(log_sum - log_bj) < 1e-3):
                num_sat_constrs += num_cur_constr
                continue

            norm_tmp = log_nz_x[:, None, :, :] - log_sum + log_bj

            # merge multiple constraints into one log_nz_x
            group_idx = torch.where(log_nz_Aj != -float('inf'))
            rearrange_idx = torch.argsort(group_idx[3])
            log_nz_x = norm_tmp[:, group_idx[1][rearrange_idx], :, group_idx[3][rearrange_idx]].permute(1, 2, 0)

            log_sum = torch.logsumexp(log_nz_x, 1, keepdim=True) # batch x 1 x n_v
            log_nz_x = log_nz_x - log_sum

            scatter_idx = nz_indices.unsqueeze(0).unsqueeze(1).repeat(batch_size, 2, 1)
            log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, :, :])

        if num_sat_constrs == num_constr:
            break

    x = torch.exp(log_x[:, 0, :num_var]) # remove dummy row & transform from log scale

    with torch.no_grad():
        cv_Ab = torch.matmul(ori_A, x.t()).t() - ori_b.unsqueeze(0)
        cv_Cd = -torch.matmul(ori_C, x.t()).t() + ori_d.unsqueeze(0)
        cv_Ef = torch.abs(torch.matmul(ori_E, x.t()).t() - ori_f.unsqueeze(0))
        if not no_warning and \
                torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) > 0.1 * batch_size:
            print('Warning: non-zero constraint violation within max iterations. Add more iterations or '
                  'the problem may be infeasible',
                  file=sys.stderr)

    return x


def linsat_kernel_grouped_sparse_v2(x, grouped_A, grouped_b, grouped_nz, tau, max_iter, dummy_val,
                                    batch_size, num_var, num_constr, ori_A, ori_b, ori_C, ori_d, ori_E, ori_f,
                                    no_warning):
    # add dummy variables
    dum_x1 = torch.full((batch_size, 2, num_constr), dummy_val, dtype=x.dtype, device=x.device)
    dum_x2 = torch.full((batch_size, num_var), dummy_val, dtype=x.dtype, device=x.device)

    # operations are performed on log scale
    log_x = x / tau
    log_dum_x1 = dum_x1 / tau
    log_dum_x2 = dum_x2 / tau

    log_gA = [torch.sparse_coo_tensor(A.indices(), torch.log(A.values()), A.shape).coalesce() for A in grouped_A]
    log_gb = [torch.log(b) for b in grouped_b]

    # perform a row norm first
    log_x = torch.stack((log_x, log_dum_x2), dim=1) # batch x 2 x n_v
    log_x = torch.cat((log_x, log_dum_x1), dim=2) # batch x 2 x (n_v + n_c)
    log_sum = torch.logsumexp(log_x, 1, keepdim=True)  # batch x 1 x (n_v + n_c)
    log_x = log_x - log_sum

    if any([torch.any(torch.isinf(log_b)) for log_b in log_gb]): raise RuntimeError('Inf encountered in log_b!')
    if any([torch.any(torch.isnan(log_A)) for log_A in log_gA]): raise RuntimeError('Nan encountered in log_A!')
    if any([torch.any(torch.isnan(log_b)) for log_b in log_gb]): raise RuntimeError('Nan encountered in log_b!')

    # Multi-set marginal Sinkhorn iterations
    for i in range(max_iter):
        num_sat_constrs = 0
        for log_nz_Aj, log_bj, nz_indices in zip(log_gA, log_gb, grouped_nz):  # log_nz_Aj is n_c x n_v; log_bj is n_c x 2
            num_cur_constr = log_nz_Aj.shape[0]

            log_x_tmp = torch.sparse_coo_tensor(
                torch.stack((
                    log_nz_Aj.indices()[0],
                    log_nz_Aj.indices()[1],
                )),  # indices
                (log_nz_Aj.values()[:, None, None].repeat(1, batch_size, 2) +
                 torch.gather(log_x.permute(2, 0, 1), 0, log_nz_Aj.indices()[1][:, None, None].repeat(1, batch_size, 2))),  # values
                (log_nz_Aj.shape[0], log_x.shape[2], batch_size, 2)  # shape
            )  # log_x + log_Aj, n_c x n_v x batch x 2 (batch, 2 are dense dimensions)

            log_sum = _logsumexp_dim(log_x_tmp, 1).permute(1, 0, 2)  # batch x n_c x 2
            if torch.all(torch.abs(log_sum - log_bj[None, :, :]) < 1e-3):
                num_sat_constrs += num_cur_constr
                continue

            norm_tmp = log_x[:, None, :, :] - log_sum[:, :, :, None] + log_bj[None, :, :, None]  # batch x n_c x 2 x n_v

            # merge multiple constraints into one log_nz_x
            rearrange_idx = torch.argsort(log_nz_Aj.indices()[1])
            log_nz_x = norm_tmp[:, log_nz_Aj.indices()[0][rearrange_idx], :, log_nz_Aj.indices()[1][rearrange_idx]].permute(1, 2, 0)

            log_sum = torch.logsumexp(log_nz_x, 1, keepdim=True) # batch x 1 x n_v
            log_nz_x = log_nz_x - log_sum

            scatter_idx = nz_indices.unsqueeze(0).unsqueeze(1).repeat(batch_size, 2, 1)
            log_x = torch.scatter(log_x, -1, scatter_idx, log_nz_x[:, :, :])

        if num_sat_constrs == num_constr:
            break

    x = torch.exp(log_x[:, 0, :num_var]) # remove dummy row & transform from log scale

    with torch.no_grad():
        cv_Ab = torch.matmul(ori_A, x.t()).t() - ori_b.unsqueeze(0)
        cv_Cd = -torch.matmul(ori_C, x.t()).t() + ori_d.unsqueeze(0)
        cv_Ef = torch.abs(torch.matmul(ori_E, x.t()).t() - ori_f.unsqueeze(0))
        if not no_warning and \
                torch.sum(cv_Ab[cv_Ab > 0]) + torch.sum(cv_Cd[cv_Cd > 0]) + torch.sum(cv_Ef[cv_Ef > 0]) > 0.1 * batch_size:
            print('Warning: non-zero constraint violation within max iterations. Add more iterations or '
                  'the problem may be infeasible',
                  file=sys.stderr)

    return x


################################################################################
#                           Helper and utils functions                         #
################################################################################

def _init_shape(mat, vec, num_var, num_constr, device, is_sparse):
    """
    Initialize and check input tensor shapes, count number of constraints
    """
    if mat is not None:
        if vec is None: raise ValueError('You must specify A-b, C-d, E-f together in pairs!')
        if vec.is_sparse: raise ValueError('Constraint can only be dense-dense or sparse-dense pairs!')
        if torch.any(vec < 0): raise ValueError('All constraints must be non-negative!')

        if is_sparse:
            if mat.layout != torch.sparse_coo: raise ValueError('Only sparse_coo Tensors are supported!')
            if torch.any(mat.values() < 0): raise ValueError('All constraints must be non-negative!')
            if len(mat.values()) == 0: raise ValueError('All-zero constraint is found!')
        else:
            if torch.any(mat < 0): raise ValueError('All constraints must be non-negative!')
            if torch.any(torch.sum(mat, dim=1) == 0): raise ValueError('All-zero constraint is found!')
        num_constr += mat.shape[0]
        if vec.shape[0] != mat.shape[0]:
            raise ValueError(f'Input shapes do not match! Got {mat.shape} and {vec.shape}')
        if mat.shape[1] != num_var:
            raise ValueError(
                f'Input shapes do not match! Got {mat.shape} but the number of variables is {num_var}.')
    else:
        if is_sparse:
            mat = torch.zeros(0, num_var, device=device, layout=torch.sparse_coo)
        else:
            mat = torch.zeros(0, num_var, device=device)
        vec = torch.zeros(0, device=device)
    return mat, vec, num_constr


def _group_constr_greedy(A, b, num_var, num_constr):
    """
    Group non-overlapping constraints in a greedy fashion
    """
    # construct group indices
    if A.is_sparse:
        constr_groups = [{'indices': [], 'nzmap': set()}]
        for i in range(num_constr):
            constr_nz = set(A[i].coalesce().indices()[0].tolist())
            inserted = False
            for group in constr_groups:
                if len(constr_nz.intersection(group['nzmap'])) > 0: # conflict
                    continue
                else:
                    group['indices'].append(i)
                    group['nzmap'] = group['nzmap'].union(constr_nz)
                    inserted = True
                    break
            if not inserted:
                constr_groups.append({'indices': [i], 'nzmap': constr_nz})
        constr_groups = [{'indices': torch.tensor(g['indices'], device=A.device),
                          'nzmap': torch.tensor(list(g['nzmap']), device=A.device)} for g in constr_groups]
    else:
        constr_groups = [{'indices': [], 'nzmap': torch.zeros(num_var + num_constr, device=A.device)}]
        for i in range(num_constr):
            constr_nz = (A[i] != 0) * 1.
            inserted = False
            for group in constr_groups:
                if torch.any(group['nzmap'] + constr_nz > 1): # conflict
                    continue
                else:
                    group['indices'].append(i)
                    group['nzmap'] += constr_nz
                    inserted = True
                    break
            if not inserted:
                constr_groups.append({'indices': [i], 'nzmap': constr_nz})
        constr_groups = [{'indices': torch.tensor(g['indices'], device=A.device),
                          'nzmap': torch.where(g['nzmap'] > 0)[0]} for g in constr_groups]

    # split A, b into groups
    grouped_A = []
    grouped_b = []
    grouped_nz = []
    if A.is_sparse:
        for group in constr_groups:
            indices = group['indices']
            nz = group['nzmap']
            grouped_A.append(torch.stack([A[i] for i in indices], dim=0).coalesce())
            grouped_b.append(b[indices])
            grouped_nz.append(nz)
    else:
        for group in constr_groups:
            indices = group['indices']
            nz = group['nzmap']
            grouped_A.append(A[indices, :][:, nz])
            grouped_b.append(b[indices])
            grouped_nz.append(nz)

    return {'grouped_A': grouped_A, 'grouped_b': grouped_b, 'grouped_nz': grouped_nz}


def _sum_dim(input, dim):
    """A sparse-friendly sum over dim function"""
    if input.is_sparse:
        if not input.is_coalesced(): input = input.coalesce()
        assert len(input.shape) == 2, 'only matrix sum is implemented for sparse tensor'
        new_val = torch.scatter_add(torch.zeros(input.shape[dim-1], device=input.device),
                                    0,
                                    input.indices()[dim-1],
                                    input.values())
    else:
        new_val = input.sum(dim=dim)
    return new_val


def _logsumexp_dim(input, dim):
    """A sparse-friendly logsumexp over dim function"""
    if input.is_sparse:
        if not input.is_coalesced(): input = input.coalesce()
        assert input.sparse_dim() == 2, 'only sparse_dim=2 is implemented'
        assert dim < input.sparse_dim()  # sparse dimensions are the first dimensions
        values = input.values()
        c = values.max()  # constant for numerical stability
        exp_v = torch.exp(values - c)  # exp

        # fit multi dense dimensions
        target_shape = [input.shape[:input.sparse_dim()][dim-1]]
        interest_indices = input.indices()[dim-1]
        for i, d in enumerate(exp_v.shape):
            if i == 0:
                continue
            target_shape.append(d)
            interest_indices = interest_indices.unsqueeze(-1).repeat_interleave(d, dim=-1)

        sum_v = torch.scatter_add(torch.zeros(target_shape, device=input.device),  # sum
                                  0,
                                  interest_indices,
                                  exp_v)
        new_val = torch.log(sum_v) + c  # log
    else:
        new_val = torch.logsumexp(input, dim=dim)
    return new_val


def _elem_div(mat, vec, dim):
    """A sparse-friendly matrix-vector element-wise division over dimension"""
    if mat.is_sparse:
        if not mat.is_coalesced(): mat = mat.coalesce()
        assert len(mat.shape) == 2, 'only matrix division is implemented for sparse tensor'
        denominators = torch.gather(vec, 0, mat.indices()[dim-1])
        new_val = mat.values() / denominators
        new_mat = torch.sparse_coo_tensor(mat.indices(), new_val, mat.shape).coalesce()
    else:
        vec = vec.unsqueeze(dim)
        new_mat = mat / vec
    return new_mat


if __name__ == '__main__':
    import pygmtools as pygm
    pygm.BACKEND = 'pytorch'
    import time
    import itertools

    # This example shows how to encode doubly-stochastic constraint for 3x3 variables
    E = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [1, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 1]], dtype=torch.float32
    )
    f = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)

    w = torch.rand(9) # w could be the output of neural network
    w = w.requires_grad_(True)

    x_gt = torch.tensor(
        [1, 0, 0,
         0, 1, 0,
         0, 0, 1], dtype=torch.float32
    )

    # Test with LinSAT
    with torch.autograd.set_detect_anomaly(True):
        prev_time = time.time()
        linsat_outp = linsat_layer(w, E=E, f=f, tau=0.1, max_iter=10, dummy_val=0, grouped=False, mode='v1')
        print(f'Ungrouped LinSAT (version<=0.0.9) forward time: {time.time() - prev_time:.4f}')
        prev_time = time.time()
        linsat_outp_g = linsat_layer(w, E=E, f=f, tau=0.1, max_iter=10, dummy_val=0, grouped=True, mode='v2')
        print(f'Grouped LinSAT (new default) forward time: {time.time() - prev_time:.4f}')
        prev_time = time.time()
        linsat_outp_gs = linsat_layer(w, E=E.to_sparse(), f=f, tau=0.1, max_iter=10, dummy_val=0, grouped=True, mode='v2')
        print(f'Grouped Sparse LinSAT forward time: {time.time() - prev_time:.4f}')
        prev_time = time.time()
        loss = ((linsat_outp - x_gt) ** 2).sum()
        loss.backward()
        print(f'LinSAT backward time: {time.time() - prev_time:.4f}')

    # Test gradient-based optimization
    niters = 10
    opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
    for i in range(niters):
        x = linsat_layer(w, E=E, f=f, tau=0.1, max_iter=10, dummy_val=0)
        cv = torch.matmul(E, x.t()).t() - f.unsqueeze(0)
        loss = ((x - x_gt) ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f'{i}/{niters}\n'
              f'  underlying obj={torch.sum(w * x)},\n'
              f'  loss={loss},\n'
              f'  sum(constraint violation)={torch.sum(cv[cv > 0])},\n'
              f'  x={x},\n'
              f'  constraint violation={cv}')
