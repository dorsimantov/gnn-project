"""
Dataset classes to handle our training and test sets.
"""

import torch
from torch_geometric.data import InMemoryDataset, Data
import os


class EXPDataset(InMemoryDataset):
    def __init__(self, root, use_new_data=False, transform=None, pre_transform=None):
        self.use_new_data = use_new_data
        super().__init__(root, transform, pre_transform)

        # Determine which files to load based on the parameter
        data_file = 'data_new.pt' if use_new_data and os.path.exists(self.processed_paths[1]) else 'data.pt'
        pre_filter_file = 'pre_filter_new.pt' if use_new_data else 'pre_filter.pt'
        pre_transform_file = 'pre_transform_new.pt' if use_new_data else 'pre_transform.pt'

        # Load the dataset files
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, data_file))

        # Optionally load pre-filter and pre-transform metadata if needed for your logic
        if os.path.exists(os.path.join(self.processed_dir, pre_filter_file)):
            self.pre_filter_data = torch.load(os.path.join(self.processed_dir, pre_filter_file))
        if os.path.exists(os.path.join(self.processed_dir, pre_transform_file)):
            self.pre_transform_data = torch.load(os.path.join(self.processed_dir, pre_transform_file))

    @property
    def raw_file_names(self):
        return ['GRAPHSAT.pkl', 'GRAPHSAT.txt']

    @property
    def processed_file_names(self):
        # Include both the original and new processed files
        return ['data.pt', 'data_new.pt', 'pre_filter.pt', 'pre_filter_new.pt',
                'pre_transform.pt', 'pre_transform_new.pt']

    def process(self):
        # This function is not invoked here as data is already processed.
        pass
