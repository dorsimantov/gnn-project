import torch
from torch_geometric.data import InMemoryDataset
import os

class EXPDataset(InMemoryDataset):
    def __init__(self):
        super().__init__('datasets/EXP')
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['GRAPHSAT.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # This function is not invoked here as data is already processed.
        pass

    def download(self):
        pass
