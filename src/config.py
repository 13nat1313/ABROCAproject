from typing import Tuple

from src.models import *
from src import torchutils

DEFAULT_CONFIGS = {
    FAST_DRO_MODEL: {
        'model_type': FAST_DRO_MODEL,
        # training parameters
        'steps': 1000,
        'batch_size': 64,
        # uncertainty set parameters
        'geometry': 'chi-square',
        'size': 0.,
        'reg': 0.0001,
        'max_iter': 1000,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'criterion_name': torchutils.FASTDRO_CRITERION,
        'momentum': 0.,
        'weight_decay': 0.0001,
        'learning_rate': 0.01,
    },
    IMPORANCE_WEIGHTING_MODEL: {
        'model_type': IMPORANCE_WEIGHTING_MODEL,
        # training parameters
        'steps': 1000,
        'batch_size': 64,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'criterion_name': torchutils.IMPORTANCE_WEIGHTING_CRITERION,
        'momentum': 0.,
        'weight_decay': 0.0001,
        'learning_rate': 0.01,
    },
    LR_MODEL: {},
    L2LR_MODEL: {},
}


def unpack_config_fastdro(config) -> Tuple[dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "geometry": config["geometry"],
                        "size": config["size"],
                        "reg": config["reg"],
                        "max_iter": config["max_iter"]}
    opt_kwargs = {"lr": config["learning_rate"],
                  "weight_decay": config["weight_decay"],
                  "momentum": config["momentum"]}
    fit_kwargs = {"steps": config["steps"],
                  "batch_size": config["batch_size"]}
    return criterion_kwargs, opt_kwargs, fit_kwargs


def unpack_config_importance_weighting(config) -> Tuple[dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"]}
    opt_kwargs = {"lr": config["learning_rate"],
                  "weight_decay": config["weight_decay"],
                  "momentum": config["momentum"]}
    fit_kwargs = {"steps": config["steps"],
                  "batch_size": config["batch_size"]}
    return criterion_kwargs, opt_kwargs, fit_kwargs


def null_config(unused_config) -> Tuple[dict, dict, dict]:
    return dict(), dict(), dict()


CONFIG_FNS = {
    FAST_DRO_MODEL: unpack_config_fastdro,
    IMPORANCE_WEIGHTING_MODEL: unpack_config_importance_weighting,
    LR_MODEL: null_config,
    L2LR_MODEL: null_config,
}
