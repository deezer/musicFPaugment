import os
from typing import Any, Dict

import GPUtil  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    output = True
    if not dist.is_available():
        output = False
    if not dist.is_initialized():
        output = False
    return output


def get_rank() -> int:
    output = 0
    if is_dist_avail_and_initialized():
        output = dist.get_rank()
    return output


def is_main_process() -> bool:
    return get_rank() == 0


def update_progress_bar(
    progress_bar: Any, loss: Dict[str, float], idx: int, t: str
) -> None:
    if is_main_process():
        for loss_type, value in loss.items():
            progress_bar.update(idx, [("{}_loss_{}".format(t, loss_type), value)])
    return


def set_gpus(max_memory: float = 0.05) -> str:
    print("TORCH version: {}".format(torch.__version__))
    print("TF version: {}".format(tf.__version__))

    #  enable CuDNN benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    gpu_index = GPUtil.getAvailable(limit=4, maxMemory=max_memory)
    # setting gpu for tensorflow
    try:
        gpu_index = [gpu_index[0]]
    except:
        raise ValueError("No GPU available!!")
    print("\t Using GPUs: {}".format(gpu_index))
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.set_visible_devices([physical_devices[gpu_index[0]]], "GPU")
    # tf.config.experimental.set_memory_growth(
    #     physical_devices[gpu_index[0]], True
    # )  # limit tf memory usage
    tf.config.set_visible_devices([], "GPU")
    device = "cuda:{}".format(",".join([str(i) for i in gpu_index]))
    return device


def fix_random_seeds(seed: int = 59) -> None:
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def remove_val_cache() -> None:
    if os.path.exists("/tmp/validation_set_cache.data-00000-of-00001"):
        os.remove("/tmp/validation_set_cache.data-00000-of-00001")
    if os.path.exists("/tmp/validation_set_cache.index"):
        os.remove("/tmp/validation_set_cache.index")
