import numpy as np
import torch

def mae(x: np.ndarray, y: np.ndarray) -> np.ndarray: 
    """Mean Absoluate Error or L1 Loss
    x: predicted value
    y: target value
    """
    return np.abs(x-y).mean(axis=0)

def rmse(x: np.ndarray, y: np.ndarray) -> np.ndarray: 
    """RMSE
    x: predicted value
    y: target value
    """
    return ((x-y) ** 2).mean(axis=0) ** (1/2)

def mape(x: np.ndarray, y: np.ndarray) -> np.ndarray: 
    """Mean Absolute Percentage Error
    x: predicted value
    y: target value
    """

    # masking 0 values in the target
    mask = y != 0.0
    count = mask.sum(0)
    # count = count if count > 0 else 1
    count[count == 0] = 1
    y = np.where(mask, y, 1)
    x = np.where(mask, x, 1)
    return np.abs((x-y)/y).sum(0) / count * 100

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


if __name__ == "__main__": 

    a = np.random.rand(207, 12)
    b = np.random.rand(207, 12)

    m1 = mae(a, b)
    m2 = rmse(a, b)
    m3 = mape(a, b)

