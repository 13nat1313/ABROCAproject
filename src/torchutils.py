import logging
import math
import time
from typing import Callable

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy
import wandb
from src.fastdro.robust_losses import RobustLoss

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def np_to_torch_float(ary) -> torch.Tensor:
    return torch.from_numpy(ary).float()


def safe_cast_to_numpy(ary):
    if isinstance(ary, np.ndarray):
        return ary
    elif isinstance(ary, torch.Tensor):
        return ary.detach().numpy()
    elif hasattr(ary, 'values'):  # covers all pandas dataframe/series types
        return ary.values
    else:
        raise NotImplementedError(f"unsupported type: {type(ary)}")


def arys_to_dataset(X, y, g, use_group=False):
    if use_group:
        return torch.utils.data.TensorDataset(
            np_to_torch_float(X),
            np_to_torch_float(y),
            np_to_torch_float(g))
    else:
        return torch.utils.data.TensorDataset(
            np_to_torch_float(X),
            np_to_torch_float(y), )


def arys_to_loader(X: pd.DataFrame, y: pd.DataFrame, g, batch_size: int,
                   shuffle=True, drop_last=False, use_group=False):
    dataset = arys_to_dataset(X, y, g, use_group=use_group)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last)
    return loader


# criterion names
DEFAULT_CRITERION = "ce"
FASTDRO_CRITERION = "fastdro"

# optimizer names
SGD_OPT = "sgd"
ADAM_OPT = "adam"


def get_optimizer(type, model, **opt_kwargs):
    logging.info(f"fetching optimizer {type} with opt_kwargs {opt_kwargs}")
    if type == SGD_OPT:
        return torch.optim.SGD(model.parameters(), **opt_kwargs)
    elif type == ADAM_OPT:
        return torch.optim.Adam(model.parameters(), **opt_kwargs)
    else:
        raise NotImplementedError


def get_criterion(criterion_name: str, **kwargs) -> Callable:
    logging.info(f"Received the following criterion parameters: {kwargs}")
    # Mean squared error
    if criterion_name == DEFAULT_CRITERION:
        return torch.nn.CrossEntropyLoss()

    # Fast DRO, using MSE_CRITERION
    elif criterion_name == FASTDRO_CRITERION:
        robust_loss = RobustLoss(
            geometry=kwargs.get('geometry', 'chi-square'),
            size=float(kwargs.get('size', 1.0)),
            reg=kwargs.get('reg', 0.01),
            max_iter=kwargs.get('max_iter', 1000)
        )

        def _loss_fn(outputs, targets):
            return robust_loss(binary_cross_entropy(outputs, targets,
                                                    reduction="none"))

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


def subgroup_loss(preds, labels, g, group_label: int) -> torch.Tensor:
    loss = binary_cross_entropy(preds, labels)
    subgroup_mask = (g == group_label).double()
    # TODO(jpgard): this should really be a masked sum.
    return torch.mean(loss * subgroup_mask)


def compute_disparity_metrics(preds, labels, sens, prefix=""):
    if prefix != "":
        prefix += "_"
    if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
        labels = np_to_torch_float(labels)
    metrics = {}
    for g in (0, 1):
        loss_g = subgroup_loss(preds, labels, sens, group_label=g)
        metrics[f"loss_{g}"] = loss_g

    metrics["loss_disparity"] = \
        metrics["loss_1"] - metrics[f"loss_0"]
    metrics["loss_abs_disparity"] = \
        abs(metrics["loss_1"] - metrics["loss_0"])
    metrics["loss_worstgroup"] = \
        max(metrics["loss_1"], metrics["loss_0"])
    return metrics


class PytorchRegressor(nn.Module):
    "A scikit-learn style interface for a linear pytorch model."

    def __init__(self, d_in: int,
                 criterion_kwargs={"criterion_name": DEFAULT_CRITERION},
                 model_type: str = "default"):
        super(PytorchRegressor, self).__init__()
        criterion_name = criterion_kwargs.pop("criterion_name")
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.fc1 = nn.Linear(d_in, 1)
        self.model_type = model_type  # used for logging

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = torch.squeeze(x)  # [batch_size,1] --> batch_size,]
        return x

    def predict(self, x):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = np_to_torch_float(x)
        return self.forward(x)

    def print_summary(self, global_step: int, metrics):
        # Print a summary every n steps
        if (global_step % 100) == 0:
            metrics_str = ', '.join(
                [f"{k}: {v}" for k, v in sorted(metrics.items())])
            logging.info(
                "metrics for model {} at step {}: {}".format(
                    self.model_type, global_step, metrics_str))

    def fit(self, X_tr, y_tr, g_tr, X_val, y_val, g_val, optimizer, steps: int,
            scheduler=None,
            batch_size=64,
            cutoff_step=1e4,
            cutoff_value=5.,
            cutoff_metric="val_loss",
            sample_weight=None):
        """

        :param X_tr: training data.
        :param y_tr: training labels.
        :param g_tr: training group labels.
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
        X_val = np_to_torch_float(X_val)
        y_val = np_to_torch_float(y_val)
        g_val = np_to_torch_float(g_val)

        if sample_weight is not None:
            raise ValueError("sample weight is not supported;"
                             "provided only for compatibility with sklearn.")

        logging.info("training pytorch model type %s for %s steps.",
                     self.model_type, steps)
        criterion = get_criterion(self.criterion_name, **self.criterion_kwargs)

        epoch = 0
        global_step = 0

        while True:
            loader = arys_to_loader(X_tr, y_tr, g_tr, batch_size=batch_size,
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
                    train_ce = binary_cross_entropy(outputs, labels,
                                                    reduction="mean")
                    val_loss = criterion(outputs_val, y_val)
                    val_ce = binary_cross_entropy(outputs_val, y_val,
                                                  reduction="mean")
                    disparity_val_metrics = compute_disparity_metrics(
                        outputs_val, y_val, g_val, "val")
                log_metrics = {
                    "train_loss": loss.item(),
                    "train_ce": train_ce.item(),
                    "val_loss": val_loss.item(),
                    "val_ce": val_ce.item(),
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
