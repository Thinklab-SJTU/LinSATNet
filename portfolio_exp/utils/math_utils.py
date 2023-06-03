import numpy as np
import torch

def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
def compute_measures(returns, annual_factor=252):
    """
    Compute mu (expected return) and cov (risk)
    :param returns: (NxF) daily returns w.r.t. to the previous day. N: number of days, F: number of assets
    :param annual_factor: compute annual return from daily return
    :return: mu, cov
    """
    assert len(returns.shape) == 2
    mu = torch.mean(returns, dim=0)
    returns_minus_mu = returns - mu.unsqueeze(0)
    cov = torch.mm(returns_minus_mu.t(), returns_minus_mu) / (returns.shape[0] - 1)
    return mu * annual_factor, cov * annual_factor


def compute_sharpe_ratio(mu, cov, weight, rf, return_details=False):
    """
    Compute the Sharpe ratio of the given portfolio
    :param mu: (F) expected return. F: number of assets
    :param cov: (FxF) risk matrix
    :param weight: (F) or (BxF) weight of assets of the portfolio. B: the batch size
    :param rf: risk-free return
    :return: the Sharpe ratio
    """
    if len(weight.shape) == 1:
        weight = weight.unsqueeze(0)
        batched_input = False
    else:
        batched_input = True
    assert len(weight.shape) == 2
    mu = mu.unsqueeze(0)
    cov = cov.unsqueeze(0)
    returns = (mu * weight).sum(dim=1)
    risk = torch.sqrt(torch.matmul(torch.matmul(weight.unsqueeze(1), cov), weight.unsqueeze(2))).squeeze()
    sharpe = (returns - rf) / risk
    if not batched_input:
        sharpe = sharpe.squeeze()
        risk = risk.squeeze()
        returns = returns.squeeze()
    if return_details:
        return sharpe, risk, returns
    else:
        return sharpe
