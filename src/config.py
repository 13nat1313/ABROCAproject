from typing import Tuple

from src.models import FAST_DRO_MODEL
from src import torchutils

DEFAULT_CONFIGS = {
    FAST_DRO_MODEL: {
        'model_type': FAST_DRO_MODEL,
        # training parameters
        'steps': 100000,
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
    }}


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


def null_config(unused_config) -> Tuple[dict, dict, dict]:
    return dict(), dict(), dict()


CONFIG_FNS = {
    FAST_DRO_MODEL: unpack_config_fastdro,
}
