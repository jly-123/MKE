import os
import re
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from torch_geometric import __version__


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def my_get_dataloader(dataset, index_list_params):
    index_list = index_list_params['index_list']
    batch_size = index_list_params['batch_size']

    train_indices = index_list[0]
    test_indices = index_list[1]
    dev_indices = index_list[2]

    train = Subset(dataset, train_indices)
    eval = Subset(dataset, dev_indices)
    test = Subset(dataset, test_indices)

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


def compatible_state_dict(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:
            comp_key = re.sub(r'conv(1|s.[0-9]).weight', r'conv\g<1>.lin.weight', key)
            if comp_key != key:
                comp_value = value.T
        if comp_key != key:
            comp_state_dict[key] = value
        comp_state_dict[comp_key] = comp_value
    return comp_state_dict


def configure_paths(config, package_root):

    def absolute(value):
        value = os.path.expanduser(str(value))
        return value if os.path.isabs(value) else os.path.join(package_root, value)

    config.models.gnn_saving_dir = absolute(config.models.gnn_saving_dir)
    config.datasets.dataset_root = absolute(config.datasets.dataset_root)
    config.explainers.explanation_result_dir = absolute(config.explainers.explanation_result_dir)
    config.record_filename = absolute(config.record_filename)


def resolve_input_csv(config, package_root):
    configured = config.get("input_csv")
    path = (os.path.expanduser(str(configured)) if configured else
            os.path.join(config.datasets.dataset_root, str(config.datasets.csv_filename)))
    if not os.path.isabs(path):
        path = os.path.join(package_root, path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Required molecular CSV not found: {}. See docs/DATASETS.md".format(path))
    return path
