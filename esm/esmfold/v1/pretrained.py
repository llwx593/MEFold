from pathlib import Path

import torch

from esm.esmfold.v1.esmfold import ESMFold
import time
import GPUtil
import logging
import threading
logging.getLogger().setLevel(logging.INFO)

class MemoryLogger:
    def __init__(self, device_id: int):
        gpu = GPUtil.getGPUs()[device_id]
        logging.info(f'Initializing: Used memory = {gpu.memoryUsed}MiB, Free memory = {gpu.memoryFree}MiB, '
                     f'Total memory = {gpu.memoryTotal}MiB')

        self.device_id = device_id
        self.currentMemoryUsed = gpu.memoryUsed
        self.allocatedMemory = gpu.memoryUsed
        self.maxMemory = 0
        self.flag = True
        
    def IntimeRecord(self):
        # gpu = GPUtil.getGPUs()[self.device_id]
        import pynvml
        pynvml.nvmlInit()
        while True:
            if not self.flag:
                print(f"trunk peak memory {self.maxMemory}")
                break
            # freeMem = gpu.memoryFree
            # usedMem = gpu.memoryUsed
            # print("!!!",usedMem - self.allocatedMemory)
            handler = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            total = round(meminfo.total / 1024 / 1024, 2)
            used = round(meminfo.used / 1024 / 1024, 2)
            free = round(meminfo.free / 1024 / 1024, 2)
            # if used - self.allocatedMemory > self.maxMemory:
            #     self.maxMemory = used - self.allocatedMemory             
            # self.currentMemoryUsed = used 
            if used > self.maxMemory:
                self.maxMemory = used
            time.sleep(0.00000000005)
        
    
    def IntimeRecordThreadStart(self):
        self.recordthread = threading.Thread(target=self.IntimeRecord,)
        self.recordthread.start()
        # self.recordthread.daemon=True
        
    def IntimeRecordThreadEnd(self):
        self.flag = False
        self.recordthread.join()
        
    def log(self, mark: str):
        # synchronize
        gpu = GPUtil.getGPUs()[self.device_id]
        freeMem = gpu.memoryFree
        usedMem = gpu.memoryUsed
        logging.info(f'{mark}: Allocate memory = {usedMem - self.allocatedMemory: .1f}MiB, '
                     f'Free memory = {freeMem: .1f}MiB, Increased memory = {usedMem - self.currentMemoryUsed: .1f}')
        self.currentMemoryUsed = usedMem
        
def _load_model(model_name):
    # memorylogger=MemoryLogger(0)
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from hub
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
        # model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="gpu")
        
    
    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    # model_state.to("cuda:0")
    # memorylogger.log("load esmfold state")
    model = ESMFold(esmfold_config=cfg)
    # memorylogger.log("esm load sate")
    
    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model


def _get_model_definition(model_name):
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from hub
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")

    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    # model.load_state_dict(model_state, strict=False)

    return model


def esmfold_v0():
    """
    ESMFold v0 model with 3B ESM-2, 48 folding blocks.
    This version was used for the paper (Lin et al, 2022). It was trained 
    on all PDB chains until 2020-05, to ensure temporal holdout with CASP14
    and the CAMEO validation and test set reported there.
    """
    return _load_model("esmfold_3B_v0")


def esmfold_v1():
    """
    ESMFold v1 model using 3B ESM-2, 48 folding blocks.
    ESMFold provides fast high accuracy atomic level structure prediction
    directly from the individual sequence of a protein. ESMFold uses the ESM2
    protein language model to extract meaningful representations from the
    protein sequence.
    """
    return _load_model("esmfold_3B_v1")

def esmfold_v1_structure():
    print("!!get structure done!!")
    return _get_model_definition("esmfold_3B_v1")