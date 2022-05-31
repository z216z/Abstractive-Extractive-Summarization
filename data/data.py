""" CNN/DM dataset"""
import json
import re
import os
from os.path import join
import tarfile
from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)
        self._file_names = os.listdir(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, self._file_names[i])) as f:
            js = json.loads(f.read())
        return js


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    def match(name): return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def extract_data(full_path):
    if os.path.exists(full_path):
        for i, file_path in enumerate(os.listdir(full_path)):
            if file_path.endswith('.tar') or file_path.endswith('.tar.gz'):
                file_path = os.path.join(full_path, file_path)
                file = tarfile.open(file_path)
                file.extractall(full_path)
                file.close()
                os.remove(file_path)
    else:
        print('The selected dataset is not available')
