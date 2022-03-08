import logging
import math
import time
from typing import Callable

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.functional import mse_loss
import wandb

from src.utils import LOG_LEVEL
from src.fastdro.robust_losses import RobustLoss

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def pd_to_torch_float(df) -> torch.Tensor:
    return torch.from_numpy(df.values).float()


def safe_cast_to_numpy(ary):
    if isinstance(ary, np.ndarray):
        return ary
    elif isinstance(ary, torch.Tensor):
        return ary.detach().numpy()
    elif hasattr(ary, 'values'):  # covers all pandas dataframe/series types
        return ary.values
    else:
        raise NotImplementedError(f"unsupported type: {type(ary)}")


def dataframes_to_dataset(X, y, use_group=False):
    if use_group:
        return torch.utils.data.TensorDataset(
            pd_to_torch_float(X),
            pd_to_torch_float(y),
            pd_to_torch_float(X['sensitive']))
    else:
        return torch.utils.data.TensorDataset(
            pd_to_torch_float(X),
            pd_to_torch_float(y))


def dataframes_to_loader(X: pd.DataFrame, y: pd.DataFrame, batch_size: int,
                         shuffle=True, drop_last=False, use_group=False):
    dataset = dataframes_to_dataset(X, y, use_group=use_group)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last)
    return loader


# criterion_name names
MSE_CRITERION = "mse"
FASTDRO_CRITERION = "fastdro"
LVR_CRITERION = "loss_variance"
CLV_CRITERION = "coarse_loss_variance"

# optimizer names
SGD_OPT = "sgd"
ADAM_OPT = "adam"


def get_criterion(criterion_name: str, **kwargs) -> Callable:
    logging.info(f"Received the following criterion parameters: {kwargs}")
    # Mean squared error
    if criterion_name == MSE_CRITERION:
        return torch.nn.MSELoss()

    # Fast DRO, using MSE_CRITERION
    elif criterion_name == FASTDRO_CRITERION:
        robust_loss = RobustLoss(
            geometry=kwargs.get('geometry', 'chi-square'),
            size=float(kwargs.get('size', 1.0)),
            reg=kwargs.get('reg', 0.01),
            max_iter=kwargs.get('max_iter', 1000)
        )

        def _loss_fn(outputs, targets):
            # taken from https://github.com/daniellevy/fast-dro/\
            # blob/dc75246ed5df5c40a54990916ec351ec2b9e0d86/train.py#L343
            return robust_loss((outputs.squeeze() - targets) ** 2)

        return _loss_fn

    # Loss variance regularization ('vanilla' loss variance),
    # using MSE_CRITERION
    elif criterion_name == LVR_CRITERION:
        loss_lambda = kwargs['lv_lambda']

        def _loss_fn(outputs, targets):
            elementwise_loss = (outputs.squeeze() - targets) ** 2
            loss_variance = torch.var(elementwise_loss)
            return torch.mean(elementwise_loss) + loss_lambda * loss_variance

        return _loss_fn
    else:
        raise NotImplementedError


def get_optimizer(type, model, **opt_kwargs):
    logging.info(f"fetching optimizer {type} with opt_kwargs {opt_kwargs}")
    if type == SGD_OPT:
        return torch.optim.SGD(model.parameters(), **opt_kwargs)
    elif type == ADAM_OPT:
        return torch.optim.Adam(model.parameters(), **opt_kwargs)
    else:
        raise NotImplementedError


def subgroup_mse(preds, labels, g, group_label: int) -> torch.Tensor:
    mse = mse_loss(preds, labels)
    subgroup_mask = (g == group_label).double()
    return torch.sum(mse * subgroup_mask)


def compute_disparity_metrics(preds, labels, sens, prefix=""):
    if prefix != "":
        prefix += "_"
    if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
        labels = pd_to_torch_float(labels)
    metrics = {}
    for g in (0, 1):
        mse_g = subgroup_mse(preds, labels, sens, group_label=g)
        metrics[f"mse_{g}"] = mse_g
        metrics[f"rmse_{g}"] = math.sqrt(mse_g)
    for metric in ("mse", "rmse"):
        metrics[f"{metric}_disparity"] = \
            metrics[f"{metric}_1"] - metrics[f"{metric}_0"]
        metrics[f"{metric}_abs_disparity"] = \
            abs(metrics[f"{metric}_1"] - metrics[f"{metric}_0"])
        metrics[f"{metric}_worstgroup"] = \
            max(metrics[f"{metric}_1"], metrics[f"{metric}_0"])
    return metrics


class PytorchRegressor(nn.Module):
    "A scikit-learn style interface for a linear pytorch model."

    def __init__(self, d_in: int,
                 criterion_kwargs={"criterion_name": MSE_CRITERION},
                 model_type: str = "default"):
        super(PytorchRegressor, self).__init__()
        criterion_name = criterion_kwargs.pop("criterion_name")
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.fc1 = nn.Linear(d_in, 1)
        self.model_type = model_type  # used for logging

    def forward(self, x):
        x = self.fc1(x)
        x = torch.squeeze(x)  # [batch_size,1] --> batch_size,]
        return x

    def predict(self, x):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = pd_to_torch_float(x)
        return self.forward(x)

    def print_summary(self, global_step: int, metrics):
        # Print a summary every n steps
        if (global_step % 100) == 0:
            metrics_str = ', '.join(
                [f"{k}: {v}" for k, v in sorted(metrics.items())])
            logging.info(
                "metrics for model {} at step {}: {}".format(
                    self.model_type, global_step, metrics_str))

    def fit(self, X_tr, y_tr, X_val, y_val, optimizer, steps: int,
            scheduler=None,
            batch_size=64,
            cutoff_step=1e4,
            cutoff_value=2e5,
            cutoff_metric="val_rmse",
            sample_weight=None):
        """

        :param X_tr: training data.
        :param y_tr: training labels.
        :param X_val: validation data.
        :param y_val: validation labels.
        :param optimizer: optimizer.
        :param steps: max number of training steps to train for.
        :param scheduler: optional learning rate scheduler.
        :param batch_size: batch size to use.
        :param cutoff_step: at this step, if val_loss ever exceeds
            cutoff_loss, terminate training.
        :param cutoff_value: threshold to terminate training
            after cutoff_step.
        :param cutoff_metric: key to use for cutoff. Must be the name of
            a logged metric.
        :param sample_weight:
        :return: None.
        """

        # TODO(jpgard): for fastdro model, add iterate averaging
        #  via fastdro.average_step(); this is described in
        #  Sec F.2, p.54 of https://arxiv.org/pdf/2010.05893.pdf

        sens_val = pd_to_torch_float(X_val["sensitive"])
        X_val = pd_to_torch_float(X_val)
        y_val = pd_to_torch_float(y_val)

        if sample_weight is not None:
            raise ValueError("sample weight is not supported;"
                             "provided only for compatibility with sklearn.")

        logging.info("training pytorch model type %s for %s steps.",
                     self.model_type, steps)
        criterion = get_criterion(self.criterion_name, **self.criterion_kwargs)

        epoch = 0
        global_step = 0

        while True:
            loader = dataframes_to_loader(X_tr, y_tr, batch_size=batch_size,
                                          use_group=True)
            for batch_idx, batch in enumerate(loader):
                cur = time.time()
                data, labels, sens = batch

                # When there is only one observation, updates can be unstable
                # due to large targets/loss values; skip these updates.
                if len(data) == 1:
                    continue

                # Model update step
                optimizer.zero_grad(set_to_none=True)
                outputs = self.forward(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Compute validation metrics
                with torch.no_grad():
                    outputs_val = self.forward(X_val)
                    train_mse = mse_loss(outputs, labels, reduction="mean")
                    val_loss = criterion(outputs_val, y_val)
                    val_mse = mse_loss(outputs_val, y_val,
                                       reduction="mean")
                    disparity_val_metrics = compute_disparity_metrics(
                        outputs_val, y_val, sens_val, "val")
                log_metrics = {
                    "train_loss": loss.item(),
                    "train_mse": train_mse.item(),
                    "train_rmse": math.sqrt(train_mse.item()),
                    "val_loss": val_loss.item(),
                    "val_mse": val_mse.item(),
                    "val_rmse": math.sqrt(val_mse.item()),
                    "step_time": time.time() - cur,
                    **disparity_val_metrics,
                }
                if scheduler is not None:
                    log_metrics["learning_rate"] = scheduler.get_last_lr()[-1]
                    # Schedule happens on per-step, not per-epoch, basis.
                    scheduler.step()
                wandb.log(log_metrics)

                self.print_summary(global_step, log_metrics)

                # Kill training if loss is nan (can occur with i.e. large
                # learning rates and/or small batch sizes), or if cutoff
                # criteria are met.
                if torch.any(torch.isnan(loss)):
                    logging.error(
                        "NaN loss detected; terminating training"
                        "{} at epoch {} step {} global_step {}: {}".format(
                            self.model_type, epoch, batch_idx, global_step,
                            log_metrics)
                    )
                    return

                # Check if loss is too high
                if (global_step >= cutoff_step) and \
                        (log_metrics[cutoff_metric] > cutoff_value):
                    logging.info(
                        "Terminating training at step "
                        "{}; loss of {} exceeds threshold of {}.".format(
                            global_step, loss.item(), cutoff_value))
                    return

                # Check if training is finished
                if global_step >= steps:
                    return

                global_step += 1
            epoch += 1
