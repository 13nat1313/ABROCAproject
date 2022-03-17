from typing import Tuple

from src.models import *
from src import torchutils

DEFAULT_CONFIGS = {
    FAST_DRO_MODEL: {
        'model_type': FAST_DRO_MODEL,
        # training parameters
        'steps': 10000,
        'batch_size': 256,
        # uncertainty set parameters
        'geometry': 'chi-square',
        'size': 0.01,
        'reg': 0.,
        'max_iter': 1000,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'criterion_name': torchutils.FASTDRO_CRITERION,
        'momentum': 0.5,
        'weight_decay': 0.,
        'learning_rate': 0.1,
    },
    DORO_MODEL: {
        'model_type': DORO_MODEL,
        # training parameters
        'steps': 10000,
        'batch_size': 64,
        # DORO criterion parameters
        'geometry': 'chi-square',
        'alpha': 0.5,  # see appendix B.3 of DORO paper
        'eps': 0.1,  # see appendix B.3 of DORO paper
        'criterion_name': torchutils.DORO_CRITERION,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'momentum': 0.5,
        'weight_decay': 0.0001,
        'learning_rate': 0.01,
    },
    GROUP_DRO_MODEL: {
        'model_type': GROUP_DRO_MODEL,
        # training parameters
        'steps': 10000,
        'batch_size': 512,
        # criterion parameters
        'criterion_name': torchutils.GROUP_DRO_CRITERION,
        'group_weights_step_size': 0.2,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'momentum': 0.99,
        'weight_decay': 0.0001,
        'learning_rate': 0.001,
    },
    IMPORANCE_WEIGHTING_MODEL: {
        'model_type': IMPORANCE_WEIGHTING_MODEL,
        # training parameters
        'steps': 10000,
        'batch_size': 128,
        # optimization parameters
        'optimizer': torchutils.SGD_OPT,
        'criterion_name': torchutils.IMPORTANCE_WEIGHTING_CRITERION,
        'momentum': 0.5,
        'weight_decay': 0.0001,
        'learning_rate': 0.001,
    },
    LR_MODEL: {
        'model_type': LR_MODEL,
    },
    LR_MODEL_BALANCED: {
        'model_type': LR_MODEL_BALANCED
    },
    L2LR_MODEL: {
        'model_type': L2LR_MODEL,
    },
    L2LR_MODEL_BALANCED: {
        'model_type': L2LR_MODEL_BALANCED,
    }
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


def unpack_config_doro(config) -> Tuple[dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "geometry": config["geometry"],
                        "alpha": config["alpha"],
                        "eps": config["eps"]}
    opt_kwargs = {"lr": config["learning_rate"],
                  "weight_decay": config["weight_decay"],
                  "momentum": config["momentum"]}
    fit_kwargs = {"steps": config["steps"],
                  "batch_size": config["batch_size"]}
    return criterion_kwargs, opt_kwargs, fit_kwargs


def unpack_config_group_dro(config) -> Tuple[dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "group_weights_step_size": config[
                            "group_weights_step_size"]}
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
    del unused_config
    return dict(), dict(), dict()


CONFIG_FNS = {
    FAST_DRO_MODEL: unpack_config_fastdro,
    DORO_MODEL: unpack_config_doro,
    GROUP_DRO_MODEL: unpack_config_group_dro,
    IMPORANCE_WEIGHTING_MODEL: unpack_config_importance_weighting,
    LR_MODEL: null_config,
    LR_MODEL_BALANCED: null_config,
    L2LR_MODEL: null_config,
    L2LR_MODEL_BALANCED: null_config,
}
