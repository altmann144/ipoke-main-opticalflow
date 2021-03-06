import torch
import os
import subprocess
import logging
import yaml
import logging.config
import inspect
from os import walk
import numpy as np
import coloredlogs
import multiprocessing as mp
from threading import Thread
from queue import Queue
from collections import abc
import cv2
from torch import nn
# import kornia


def get_member(model, name):
    if isinstance(model, nn.DataParallel):
        module = model.module
    else:
        module = model

    return getattr(module, name)

def preprocess_image(img,swap_channels=False):
    """

    :param img: numpy array of shape (H,W,3)
    :param swap_channels: True, if channelorder is BGR
    :return:
    """
    if swap_channels:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # this seems to be possible as flownet2 outputs only images which can be divided by 64
    shape = img.shape
    img = img[:int(shape[0] / 64) * 64,:int(shape[1] / 64) * 64]

    return img

class LoggingParent:
    def __init__(self):
        super().__init__()
        # find project root
        mypath = inspect.getfile(self.__class__)
        mypath = "/".join(mypath.split("/")[:-1])
        found = False
        while mypath!="" and not found:
            f = []
            for (dirpath, dirnames, filenames) in walk(mypath):
                f.extend(filenames)
                break
            if ".gitignore" in f:
                found = True
                continue
            mypath = "/".join(mypath.split("/")[:-1])
        project_root = mypath+"/"
        # Put it together
        file = inspect.getfile(self.__class__).replace(project_root, "").replace("/", ".").split(".py")[0]
        cls = str(self.__class__)[8:-2]
        cls = str(cls).replace("__main__.", "").split(".")[-1]
        self.logger = get_logger()

def get_gpu_id_with_lowest_memory(index=0, target_gpus:list=None):
    # get info from nvidia-smi
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]

    # get the one with the lowest usage
    if target_gpus is None:
        indices = np.argsort(gpu_memory)
    else:
        indices = [i for i in np.argsort(gpu_memory) if i in target_gpus]
    return torch.device(f"cuda:{indices[-index-1]}")


iuhihfie_logger_loaded = False
def get_logger_old(name):
    # setup logging
    global iuhihfie_logger_loaded
    if not iuhihfie_logger_loaded:
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/logging.yaml', 'r') as f:
            log_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            logging.config.dictConfig(log_cfg)
            iuhihfie_logger_loaded = True
    logger = logging.getLogger(name)
    coloredlogs.install(logger=logger, level="DEBUG")
    return logger


def get_logger():
    logger = logging.getLogger("pytorch_lightning.core")
    logger.setLevel('DEBUG')
    # coloredlogs.install(logger=logger, level='DEBUG')
    return logger


def save_model_to_disk(path, models, epoch):
    for i, model in enumerate(models):
        tmp_path = path
        if not os.path.exists(path):
            os.makedirs(path)
        tmp_path = tmp_path + f"model_{i}-epoch{epoch}"
        torch.save(model.state_dict(), tmp_path)


def _do_parallel_data_prefetch(func, Q, data, idx):
    # create dummy dataset instance

    # run prefetching

    res = func(*data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
    func: callable, data, n_proc, target_data_type="ndarray",cpu_intensive=True
):
    static_args = None
    if target_data_type not in ["ndarray", "list"]:
        raise ValueError(
            "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
        )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable) and not isinstance(data,tuple):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    elif isinstance(data,tuple):
        static_args = data[1:]
        data = data[0]
        print('Using static args.')
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q,  (part,) if static_args is None else (part,*static_args), i]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, (part,) if static_args is None else (part,*static_args), i]
            for i, part in enumerate(
                [data[i : i + step] for i in range(0, len(data), step)]
            )
        ]


    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()


        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if not isinstance(gather_res[0], np.ndarray):
        return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

    # order outputs
    return np.concatenate(gather_res, axis=0)

def linear_var(
    act_it, start_it, end_it, start_val, end_val, clip_min, clip_max
):
    act_val = (
        float(end_val - start_val) / (end_it - start_it) * (act_it - start_it)
        + start_val
    )
    return np.clip(act_val, a_min=clip_min, a_max=clip_max)

def get_patches(seq_batch,weights,config,fg_value, logger = None):
    """

    :param seq_batch: Batch of videos
    :param weights: batch of flow weights for the videos
    :param config: config, containing spatial_size
    :param fg_value: foreground value of the weight map
    :return:
    """
    import kornia
    weights_as_bool = torch.eq(weights,fg_value)
    cropped = []
    for vid,weight in zip(seq_batch,weights_as_bool):
        vid_old = vid
        weight_ids = torch.nonzero(weight,as_tuple=True)
        try:
            min_y = weight_ids[0].min()
            max_y = weight_ids[0].max()
            min_x = weight_ids[1].min()
            max_x = weight_ids[1].max()
            vid = vid[...,min_y:max_y,min_x:max_x]
            if len(vid.shape) < 4:
                data_4d = vid[None,...]
                vid = kornia.transform.resize(data_4d, config["spatial_size"])
                cropped.append(vid.squeeze(0))
            else:
                vid = kornia.transform.resize(vid,config["spatial_size"])
                cropped.append(vid)
        except Exception as e:
            if logger is None:
                print(e)
            else:
                logger.warn(f'Catched the following exception in "get_patches": {e.__class__.__name__}: {e}. Skip patching this sample...')
            cropped.append(vid_old)


    return torch.stack(cropped,dim=0)


if __name__ == "__main__":
    print(get_gpu_id_with_lowest_memory())
