import torch
from torch_geometric.data import InMemoryDataset
import os

class EXPDataset(InMemoryDataset):
    def __init__(self, use_new_data=True, transform=None, pre_transform=None):
        self.use_new_data = use_new_data
        super().__init__('datasets/EXP', transform, pre_transform)

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


class CEXPDataset(InMemoryDataset):
    def __init__(self, use_new_data=False, transform=None, pre_transform=None):
        self.use_new_data = use_new_data
        super().__init__('datasets/CEXP', transform, pre_transform)

        # Determine which files to load based on the parameter
        data_file = 'data_new.pt' if use_new_data and os.path.exists(self.processed_paths[1]) else 'data.pt'
        pre_filter_file = 'pre_filter_new.pt' if use_new_data else 'pre_filter.pt'
        pre_transform_file = 'pre_transform_new.pt' if use_new_data else 'pre_transform.pt'
        data_dict_file = 'data_dict.pt'  # Add the new file

        # Load the dataset files
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, data_file))

        # Optionally load pre-filter and pre-transform metadata if needed for your logic
        if os.path.exists(os.path.join(self.processed_dir, pre_filter_file)):
            self.pre_filter_data = torch.load(os.path.join(self.processed_dir, pre_filter_file))
        if os.path.exists(os.path.join(self.processed_dir, pre_transform_file)):
            self.pre_transform_data = torch.load(os.path.join(self.processed_dir, pre_transform_file))

        # Load the data_dict file
        if os.path.exists(os.path.join(self.processed_dir, data_dict_file)):
            self.data_dict = torch.load(os.path.join(self.processed_dir, data_dict_file))
        else:
            self.data_dict = None  # or handle it appropriately if it's missing

    @property
    def raw_file_names(self):
        return ['GRAPHSAT.pkl', 'GRAPHSAT.txt']

    @property
    def processed_file_names(self):
        # Include both the original and new processed files
        return ['data.pt', 'data_new.pt', 'pre_filter.pt', 'pre_filter_new.pt',
                'pre_transform.pt', 'pre_transform_new.pt', 'data_dict.pt']

    def process(self):
        # This function is not invoked here as data is already processed.
        pass

    @property
    def len(self):
        # Make sure len is correctly defined
        return len(self.data)