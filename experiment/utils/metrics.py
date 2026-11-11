import torch


def mae(pred, target):

    return torch.mean(torch.abs(pred - target))


def mse(pred, target):

    return torch.mean((pred - target) ** 2)


def rmse(pred, target):

    return torch.sqrt(mse(pred, target))


def mape(pred, target):

    eps = 1e-5

    return torch.mean(
        torch.abs((target - pred) / (target + eps))
    ) * 100

def evaluate(pred, target):
    return {
        "MAE": mae(pred, target).item(),
        "RMSE": rmse(pred, target).item(),
        "MAPE": mape(pred, target).item(),
    }