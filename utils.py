import os
import numpy as np
from pathlib import Path

import torch


def create_folder_ifnotexist(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=False)
    return folder_path


class Tracker(object):
    
    def __init__(self):
        self.infos = {}
    
    def write_info(self, key, value):
        self.infos[key] = value
    
    def export_info(self):
        return self.infos
    
    def clean_info(self):
        self.infos = {}


def save_checkpoint(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def flatten(x, dim):
    return x.reshape(x.size()[:dim] + (-1,))


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def get_data_dict(dataloader):
    data_dict = dataloader.__next__()
    return data_dict

def get_next_batch(data_dict, test_interp=False):
    """
    获取并处理下一批次数据，确保所有张量在正确的设备上
    
    Args:
        data_dict: 包含原始数据的字典
        test_interp: 是否为插值测试模式
    Returns:
        batch_dict: 处理后的数据字典
    """
    # 首先确定主设备
    device = next(tensor for tensor in data_dict.values() 
                 if isinstance(tensor, torch.Tensor)).device
    
    # 创建新的批次字典并确保设备一致性
    batch_dict = get_dict_template()
    
    # 复制基本属性
    batch_dict["mode"] = data_dict["mode"]
    
    # 将所有张量数据转移到正确的设备
    for key in ["observed_data", "observed_tp", "data_to_predict", "tp_to_predict"]:
        if key in data_dict and isinstance(data_dict[key], torch.Tensor):
            batch_dict[key] = data_dict[key].to(device)
    
    # 处理观察数据的掩码
    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"].to(device)
        filter_mask = batch_dict["observed_mask"].unsqueeze(-1).unsqueeze(-1).to(device)
        
        if not test_interp:
            batch_dict["observed_data"] = filter_mask * batch_dict["observed_data"]
        else:
            # 插值测试模式的特殊处理
            selected_mask = batch_dict["observed_mask"].squeeze(-1).bool()  # 改用 bool() 替代 byte()
            b, t, c, h, w = batch_dict["observed_data"].size()
            batch_dict["observed_data"] = batch_dict["observed_data"][selected_mask, ...].view(b, t // 2, c, h, w)
            batch_dict["observed_mask"] = torch.ones(b, t // 2, 1).to(device)  # 确保设备一致

    # 处理预测数据的掩码
    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"].to(device)
        filter_mask = batch_dict["mask_predicted_data"].unsqueeze(-1).unsqueeze(-1).to(device)
        
        if not test_interp:
            batch_dict["orignal_data_to_predict"] = batch_dict["data_to_predict"].clone()
            batch_dict["data_to_predict"] = filter_mask * batch_dict["data_to_predict"]
        else:
            b, t, c, h, w = batch_dict["data_to_predict"].size()
            # 生成预测时间步并确保在正确的设备上
            batch_dict["tp_to_predict"] = torch.arange(0, t, device=device).float() / t
            
            # 处理掩码
            selected_mask = (torch.ones_like(batch_dict["mask_predicted_data"]) - 
                           batch_dict["mask_predicted_data"])
            selected_mask[:, -1, :] = 0.  # 排除最后一帧
            selected_mask = selected_mask.squeeze(-1).bool()  # 使用 bool() 替代 byte()
            batch_dict["mask_predicted_data"] = selected_mask
            
    return batch_dict


def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def reverse_time_order(tensor):
    idx = [i for i in range(tensor.size(1) - 1, -1, -1)]
    return tensor[:, idx, ...]


def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None
            }


def split_data_extrap(data_dict, opt):
    
    n_observed_tp = data_dict["data"].size(1) // 2
    
    split_dict = {"observed_data": data_dict["data"][:, :n_observed_tp, :].clone(),
                  "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
                  "data_to_predict": data_dict["data"][:, n_observed_tp:, :].clone(),
                  "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone(),
                  "observed_mask": None, "mask_predicted_data": None}
    
    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()
    
    split_dict["mode"] = "extrap"
        
    return split_dict


def split_data_interp(data_dict, opt):

    split_dict = {"observed_data": data_dict["data"].clone(),
                  "observed_tp": data_dict["time_steps"].clone(),
                  "data_to_predict": data_dict["data"].clone(),
                  "tp_to_predict": data_dict["time_steps"].clone(),
                  "observed_mask": None,
                  "mask_predicted_data": None}

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()
    
    split_dict["mode"] = "interp"
    
    return split_dict


def add_mask(data_dict):
    """
    为数据添加掩码，确保掩码在正确的设备上
    
    Args:
        data_dict: 包含数据的字典
    Returns:
        data_dict: 添加掩码后的字典
    """
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]
    
    if mask is None:
        # 创建掩码时直接使用数据的设备
        mask = torch.ones_like(data, device=data.device)
    else:
        # 确保掩码在正确的设备上
        mask = mask.to(data.device)
    
    data_dict["observed_mask"] = mask
    return data_dict

def split_and_subsample_batch(data_dict, opt, data_type="train"):
    if data_type == "train":
        # Training set
        if opt.extrap:
            processed_dict = split_data_extrap(data_dict, opt)
        else:
            processed_dict = split_data_interp(data_dict, opt)
    
    else:
        # Test set
        if opt.extrap:
            processed_dict = split_data_extrap(data_dict, opt)
        else:
            processed_dict = split_data_interp(data_dict, opt)
    
    # add mask
    processed_dict = add_mask(processed_dict)
    
    return processed_dict