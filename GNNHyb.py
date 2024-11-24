import numpy as np
import argparse
import torch
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch_geometric.data import Batch
from torch_scatter import scatter_max
from torch import nn
import torch_geometric.transforms as T
from k_gnn import GraphConv
import csv
from os import path
import sys

from PlanarSATPairsDataset import PlanarSATPairsDataset
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()
parser.add_argument("--no-train", default=False)
parser.add_argument("-layers", type=int, default=8)  # Number of GNN layers
# Dimensionality of GNN embeddings
parser.add_argument("-width", type=int, default=64)
# Number of training epochs
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-dataset", type=str, default="EXP")  # Dataset being used
parser.add_argument(
    "-randomRatio", type=float, default=0.0
)  # Random ratio: 1.0 -full random, 0 - deterministic
# parser.add_argument('-clip', type=float, default=0.5)    # Gradient Clipping: Disabled
parser.add_argument(
    "-probDist", type=str, default="n"
)  # Probability disttribution to initialise randomly
# n: Gaussian, xn: Xavier Gaussian, u: Uniform, xu: Xavier uniform
parser.add_argument(
    "-normLayers", type=int, default=1
)  # Normalise Layers in the GNN (default True/1)
parser.add_argument("-activation", type=str, default="tanh")  # Non-linearity used
parser.add_argument("-learnRate", type=float, default=0.00065)  # Learning Rate
parser.add_argument("-learnRateGIN", type=float, default=0.00035)  # Learning Rate
parser.add_argument(
    "-additionalRandomFeatures", type=int, default=1
)  # Additional Random Features
parser.add_argument("-convType", type=str, default="")
args = parser.parse_args()


def graph_mixup(data1, data2, alpha=0.2):
    """
    Applies mixup between two graphs.
    Args:
        data1, data2: PyTorch Geometric `Data` objects.
        alpha: Mixup interpolation factor (default 0.2).
    Returns:
        Mixed `Data` object.
    """
    mix = torch.rand(1) * alpha
    data_mix = data1.clone()
    data_mix.x = mix * data1.x + (1 - mix) * data2.x  # Interpolate node features

    # Combine edges (keep simple union for edge_index)
    edge_index = torch.cat((data1.edge_index, data2.edge_index), dim=1)
    data_mix.edge_index = torch.unique(edge_index, dim=1)  # Remove duplicate edges

    return data_mix


def drop_edges(data, p=0.1):
    """
    Randomly drops edges from the graph.
    Args:
        data: PyTorch Geometric `Data` object.
        p: Probability of dropping each edge (default 0.1).
    Returns:
        Augmented `Data` object with fewer edges.
    """
    num_edges = data.edge_index.size(1)
    mask = torch.rand(num_edges) > p  # Keep edges with probability (1 - p)
    data.edge_index = data.edge_index[:, mask]
    return data


def add_edges(data, num_new_edges=5):
    """
    Randomly adds edges to the graph.
    Args:
        data: PyTorch Geometric `Data` object.
        num_new_edges: Number of new edges to add (default 5).
    Returns:
        Augmented `Data` object with additional edges.
    """
    num_nodes = data.num_nodes
    new_edges = torch.randint(0, num_nodes, (2, num_new_edges))  # Random node pairs
    data.edge_index = torch.cat((data.edge_index, new_edges), dim=1)
    return data


def drop_nodes(data, p=0.1):
    """
    Randomly drops nodes from the graph.
    Args:
        data: PyTorch Geometric `Data` object.
        p: Probability of dropping each node (default 0.1).
    Returns:
        Augmented `Data` object with fewer nodes and edges.
    """
    num_nodes = data.num_nodes
    mask = torch.rand(num_nodes) > p  # Keep nodes with probability (1 - p)
    keep_nodes = mask.nonzero(as_tuple=True)[0]

    # Update node features
    data.x = data.x[keep_nodes]

    # Update edge index
    node_map = torch.full((num_nodes,), -1, dtype=torch.long)
    node_map[keep_nodes] = torch.arange(keep_nodes.size(0))
    mask_edges = mask[data.edge_index[0]] & mask[data.edge_index[1]]  # Remove edges of dropped nodes
    data.edge_index = node_map[data.edge_index[:, mask_edges]]  # Map old indices to new ones

    return data


def mask_features(data, mask_prob=0.1):
    """
    Masks features in the graph.
    Args:
        data: PyTorch Geometric `Data` object.
        mask_prob: Probability of masking each feature (default 0.1).
    Returns:
        Augmented `Data` object with masked features.
    """
    # Create a random mask for all features
    mask = torch.rand(data.x.size()) > mask_prob
    data.x = data.x * mask
    return data


def augment_batch(batch, augmentation_fn, **kwargs):
    """
    Applies an augmentation function to each graph in a batch.
    Args:
        batch: PyTorch Geometric `Batch` object.
        augmentation_fn: Function to apply to each graph (`Data` object).
        **kwargs: Additional arguments for the augmentation function.
    Returns:
        Augmented `Batch` object.
    """
    # Split the batch into individual graphs
    graphs = batch.to_data_list()

    # Apply the augmentation function to each graph
    augmented_graphs = [augmentation_fn(graph, **kwargs) for graph in graphs]

    # Reassemble the graphs into a new batch
    return Batch.from_data_list(augmented_graphs)


# Permute a single graph
def permute_graph(graph):
    perm = torch.randperm(graph.x.size()[0])
    graph.x = graph.x[perm]
    node_mapping = {old: new for new, old in enumerate(perm.tolist())}
    graph.edge_index = torch.tensor([[node_mapping[s.item()] for s in row] for row in graph.edge_index], dtype=torch.long)
    return graph


class PermutedDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def permute_dataset(dataset):
    permuted_data_list = []
    for graph in dataset:
        permuted_graph = permute_graph(graph.clone())
        permuted_data_list.append(permuted_graph)
    return PermutedDataset(permuted_data_list)


# Function to permute a batch
def permute_batch(data_batch):
    """
    Apply a random permutation to each graph in a batch.

    Args:
        data_batch (torch_geometric.data.Batch): Batch of graphs.

    Returns:
        torch_geometric.data.Batch: Batch with permuted nodes.
    """
    permuted_graphs = []

    for data in data_batch.to_data_list():
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)  # Generate a random permutation

        # Permute node features
        data.x = data.x[perm]

        # Permute edge_index
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(num_nodes)
        data.edge_index = inverse_perm[data.edge_index]

        # Add the permuted graph to the list
        permuted_graphs.append(data)

    # Recreate the batch with the permuted graphs
    return data_batch.__class__.from_data_list(permuted_graphs)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


def print_or_log(input_data, log=False, log_file_path="Debug.txt"):
    if not log:  # If not logging, we should just print
        print(input_data)
    else:  # Logging
        log_file = open(log_file_path, "a+")
        log_file.write(str(input_data) + "\r\n")
        log_file.close()  # Keep the file available throughout execution


class MyFilter(object):
    def __call__(self, data):
        return True  # No Filtering


class MyPreTransform(object):
    def __call__(self, data):
        data.x = F.one_hot(data.x[:, 0], num_classes=2).to(
            torch.float
        )  # Convert node labels to one-hot
        return data


# Command Line Arguments
DATASET = args.dataset
LAYERS = args.layers
EPOCHS = args.epochs
WIDTH = args.width
RANDOM_RATIO = args.randomRatio
DISTRIBUTION = args.probDist
ACTIVATION = F.elu if args.activation == "elu" else F.tanh
# CLIP = args.clip
LEARNING_RATE = args.learnRate
LEARNING_RATE_GIN = args.learnRateGIN
ADDITIONAL_RANDOM_FEATURES = args.additionalRandomFeatures
CONV_TYPE = args.convType

NORM = args.normLayers == 1
MODEL = (
    "GNNHyb-"
    + str(args.activation)
    + "-"
    + str(RANDOM_RATIO).replace(".", ",")
    + "-"
    + str(DISTRIBUTION)
    + "-"
    + str(NORM)
    + "-"
)

if LEARNING_RATE != 0.001:
    MODEL = MODEL + "lr" + str(LEARNING_RATE) + "-"

BATCH = 400
MODULO = 4
MOD_THRESH = 1

dataset = PlanarSATPairsDataset(
    root="Data/" + DATASET,
    pre_transform=T.Compose([MyPreTransform()]),
    pre_filter=MyFilter(),
)
dataset = TUDataset(root='./Data/', name='NCI1')
from collections import Counter
labels = [data.y.item() for data in dataset]

class_counts = Counter(labels)
print(f"Class distribution: {class_counts}")

# Calculate class imbalance ratio
total_samples = sum(class_counts.values())
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples, {count / total_samples:.2%} of total")



csv_file_path = (
    "log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ","
    + str(CONV_TYPE)
    + ","
    + str(ADDITIONAL_RANDOM_FEATURES)
    + ".csv"
)
if path.exists(csv_file_path):
    print(f"The file '{csv_file_path}' already exists. Stopping the script.")
    sys.exit(1)


def log_to_csv(csv_data, csv_file_path="logs.csv", headers=None):
    """
    Logs data to a CSV file.

    Parameters:
        csv_data (dict or list): The data to log, either as a dictionary or a list.
        csv_file_path (str): Path to the CSV file.
        headers (list): Optional headers for the CSV file (used only on first write).
    """
    write_headers = headers is not None and not path.exists(csv_file_path)
    mode = "a" if not write_headers else "w"

    with open(csv_file_path, mode, newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_headers:
            writer.writerow(headers)
        if isinstance(csv_data, dict):
            writer.writerow(csv_data.values())
        elif isinstance(csv_data, list):
            writer.writerow(csv_data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_type = CONV_TYPE  # Flag to control which layer type to use
        self.norm = NORM
        self.width = WIDTH
        deterministic_dims = WIDTH - int(RANDOM_RATIO * WIDTH)

        if deterministic_dims > 0:
            self.conv1 = self._get_conv_layer(dataset.num_features, 32)
            print(f"conv1 #params: {sum(p.numel() for p in self.conv1.parameters())}")
            self.conv2 = self._get_conv_layer(32, deterministic_dims)
            print(f"conv2 #params: {sum(p.numel() for p in self.conv2.parameters())}")

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(LAYERS):
            self.conv_layers.append(
                self._get_conv_layer(
                    WIDTH + ADDITIONAL_RANDOM_FEATURES,
                    WIDTH + ADDITIONAL_RANDOM_FEATURES,
                )
            )
        print(
            f"additional layers #params: {
                sum(p.numel() for p in self.conv_layers.parameters())
            }"
        )

        self.fc1 = torch.nn.Linear(
            WIDTH + ADDITIONAL_RANDOM_FEATURES, WIDTH + ADDITIONAL_RANDOM_FEATURES
        )
        self.fc2 = torch.nn.Linear(WIDTH + ADDITIONAL_RANDOM_FEATURES, 32)
        self.fc3 = torch.nn.Linear(32, dataset.num_classes)

    def _get_conv_layer(self, in_channels, out_channels):
        """
        Returns the appropriate convolutional layer based on the conv_type flag.
        """
        if self.conv_type == "gatconv":
            return GATConv(in_channels, out_channels)
        elif self.conv_type == "gcnconv":
            return GCNConv(in_channels, out_channels)
        elif self.conv_type == "sageconv":
            return SAGEConv(in_channels, out_channels)
        elif self.conv_type == "ginconv":
            return GINConv(
                nn=SimpleMLP(
                    input_dim=in_channels,
                    hidden_dim=3 * in_channels,  # A standard choice for hidden layers
                    output_dim=out_channels,
                ),
            )
        else:
            print("yo")
            # Default to GraphConv
            return GraphConv(in_channels, out_channels, norm=NORM)

    def reset_parameters(self):
        for name, module in self._modules.items():
            if hasattr(module, "reset_parameters"):
                # If the module has a reset_parameters method, call it
                module.reset_parameters()
            elif isinstance(module, (list, torch.nn.ModuleList)):
                # If it's a list or ModuleList, iterate over its elements
                for submodule in module:
                    if hasattr(submodule, "reset_parameters"):
                        print("sub: ", submodule)
                        submodule.reset_parameters()
            elif isinstance(module, int):
                # Skip integers explicitly
                print(
                    f"Skipping reset_parameters for {name} (int type detected: {
                        module
                    })"
                )
            else:
                # Handle unexpected types gracefully
                print(f"Unexpected type for {name}: {type(module)}. Skipping.")

    def forward(self, data):
        if int(RANDOM_RATIO * WIDTH) > 0:  # Randomness Exists
            random_dims = torch.empty(
                data.x.shape[0], int(RANDOM_RATIO * WIDTH)
            )  # Random INIT
            if DISTRIBUTION == "n":
                torch.nn.init.normal_(random_dims)
            elif DISTRIBUTION == "u":
                torch.nn.init.uniform_(random_dims, a=-1.0, b=1.0)
            elif DISTRIBUTION == "xn":
                torch.nn.init.xavier_normal_(random_dims)
            elif DISTRIBUTION == "xu":
                torch.nn.init.xavier_uniform_(random_dims)
            if int(RANDOM_RATIO * WIDTH) < WIDTH:  # Not Full Randomness
                data.x1 = ACTIVATION(self.conv1(data.x, data.edge_index))
                data.x2 = ACTIVATION(self.conv2(data.x1, data.edge_index))
                data.x3 = torch.cat((data.x2, random_dims), dim=1)
            else:  # Full Randomness
                data.x3 = random_dims
        else:  # No Randomness
            data.x1 = ACTIVATION(self.conv1(data.x, data.edge_index))
            data.x3 = ACTIVATION(self.conv2(data.x1, data.edge_index))
        # Add UIDs after embedding
        if ADDITIONAL_RANDOM_FEATURES > 0:
            random_dims = torch.empty(
                data.x.shape[0], ADDITIONAL_RANDOM_FEATURES
            )  # Random INIT
            if DISTRIBUTION == "n":
                torch.nn.init.normal_(random_dims)
            elif DISTRIBUTION == "u":
                torch.nn.init.uniform_(random_dims, a=-1.0, b=1.0)
            elif DISTRIBUTION == "xn":
                torch.nn.init.xavier_normal_(random_dims)
            elif DISTRIBUTION == "xu":
                torch.nn.init.xavier_uniform_(random_dims)
            data.x3 = torch.cat((data.x3, random_dims), dim=1)

        for layer in range(
            LAYERS
        ):  # Number of message passing iterations we want to test over
            data.x3 = ACTIVATION(self.conv_layers[layer](data.x3, data.edge_index))
        x = data.x3
        x = scatter_max(x, data.batch, dim=0)[0]

        if args.no_train:
            x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


device = "cpu"
conv_type = "ginconv"
model = Net().to(device)


def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        # TODO: FOR PAPER - AUGMENTATIONS FAILED
        # Apply edge dropping to the batch
        augmented_batch = augment_batch(data, drop_edges, p=0.00)
        # Apply feature masking to the batch
        augmented_batch = augment_batch(augmented_batch, mask_features, mask_prob=0.0)
        data = augmented_batch.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction="sum").item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        nb_trials = 1  # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(data.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(data).max(1)[1]
            successful_trials += pred.eq(data.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)


def compute_permutation_loss(loader):
    permuted_loss_all = 0
    for data in loader:
        with torch.no_grad():
            # print("orig", data.x, "permutated", permute_batch(data).x)
            permuted_data = permute_batch(data).to(device)
            permuted_out = model(permuted_data)
            permuted_loss = F.nll_loss(permuted_out, permuted_data.y)
            permuted_loss_all += data.num_graphs * permuted_loss.item()

    return permuted_loss_all / len(loader.dataset)


def compute_permutation_accuracy(loader):
    model.eval()
    correct = 0

    for data in loader:
        permuted_data = permute_batch(data).to(device)
        nb_trials = 1  # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(permuted_data.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(permuted_data).max(1)[1]
            successful_trials += pred.eq(permuted_data.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)


lr = LEARNING_RATE if conv_type != "ginconv" else LEARNING_RATE_GIN
acc = []
tr_acc = []
SPLITS = 1
tr_accuracies = np.zeros((EPOCHS, SPLITS))
tst_accuracies = np.zeros((EPOCHS, SPLITS))
tst_exp_accuracies = np.zeros((EPOCHS, SPLITS))
tst_lrn_accuracies = np.zeros((EPOCHS, SPLITS))

perm_tr_accuracies = np.zeros((EPOCHS, SPLITS))
perm_tst_accuracies = np.zeros((EPOCHS, SPLITS))
perm_tr_losses = np.zeros((EPOCHS, SPLITS))
perm_tst_losses = np.zeros((EPOCHS, SPLITS))
SPLITS = 5

for i in range(SPLITS):
    if i > 0:
        break
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=lr
    )

    n = len(dataset) // SPLITS
    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_exp_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_lrn_mask = torch.zeros(len(dataset), dtype=torch.bool)

    test_mask[i * n : (i + 1) * n] = 1  # Now set the masks
    learning_indices = [
        x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % MODULO <= MOD_THRESH
    ]
    test_lrn_mask[learning_indices] = 1
    exp_indices = [
        x for idx, x in enumerate(range(n * i, n * (i + 1))) if x % MODULO > MOD_THRESH
    ]
    test_exp_mask[exp_indices] = 1

    # Now load the datasets
    test_dataset = dataset[test_mask]
    permuted_test_dataset = permute_dataset(test_dataset)
    test_exp_dataset = dataset[test_exp_mask]
    test_lrn_dataset = dataset[test_lrn_mask]
    train_dataset = dataset[~test_mask]
    permuted_train_dataset = permute_dataset(train_dataset)

    n = len(train_dataset) // SPLITS
    val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    val_mask[i * n : (i + 1) * n] = 1
    val_dataset = train_dataset[val_mask]
    train_dataset = train_dataset[~val_mask]

    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    test_exp_loader = DataLoader(
        test_exp_dataset, batch_size=BATCH
    )  # These are the new test splits
    test_lrn_loader = DataLoader(test_lrn_dataset, batch_size=BATCH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    permuted_train_loader = DataLoader(permuted_train_dataset, batch_size=BATCH, shuffle=True)
    permuted_test_loader = DataLoader(permuted_test_dataset, batch_size=BATCH)

    print_or_log(
        "---------------- Split {} ----------------".format(i),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )
    best_val_loss, test_acc = 100, 0
    for epoch in range(EPOCHS):
        lr = scheduler.optimizer.param_groups[0]["lr"]
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        perm_train_acc = compute_permutation_accuracy(permuted_train_loader)
        perm_test_acc = compute_permutation_accuracy(permuted_test_loader)
        perm_train_loss = compute_permutation_loss(permuted_train_loader)
        perm_test_loss = compute_permutation_loss(permuted_test_loader)
        test_exp_acc = test(test_exp_loader)
        test_lrn_acc = test(test_lrn_loader)
        perm_tr_accuracies[epoch, i] = perm_train_acc
        perm_tst_accuracies[epoch, i] = perm_test_acc
        tr_accuracies[epoch, i] = train_acc
        tst_accuracies[epoch, i] = test_acc
        perm_tr_losses[epoch, i] = perm_train_loss
        perm_tst_losses[epoch, i] = perm_test_loss
        tst_exp_accuracies[epoch, i] = test_exp_acc
        tst_lrn_accuracies[epoch, i] = test_lrn_acc
        # Log to CSV
        data_row = [
            epoch + 1,
            lr,
            train_loss,
            val_loss,
            test_acc,
            test_exp_acc,
            test_lrn_acc,
            train_acc,
            perm_train_loss,
            perm_test_loss,
            perm_train_acc,
            perm_test_acc,
        ]
        if epoch == 0:  # Write headers only once
            headers = [
                "Epoch",
                "Learning Rate",
                "Train Loss",
                "Validation Loss",
                "Test Accuracy",
                "Test Exp Accuracy",
                "Test Learn Accuracy",
                "Train Accuracy",
                "Perm Train Loss",
                "Perm Test Loss",
                "Perm Train Accuracy",
                "Perm Test Accuracy",
            ]
            log_to_csv(data_row, csv_file_path, headers=headers)
        else:
            log_to_csv(data_row, csv_file_path)
        print_or_log(
            "Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, "
            "Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}, "
            "Perm Train Loss: {:.7f}, Perm Test Loss: {:.7f}, Perm Train Acc: {:.7f}, Perm Test Acc: {:.7f}".format(
                epoch + 1,
                lr,
                train_loss,
                val_loss,
                test_acc,
                test_exp_acc,
                test_lrn_acc,
                train_acc,
                perm_train_loss,
                perm_test_loss,
                perm_train_acc,
                perm_test_acc,
            ),
            log_file_path="log"
            + MODEL
            + DATASET
            + ","
            + str(LAYERS)
            + ","
            + str(WIDTH)
            + ".txt",
        )
    acc.append(test_acc)
    tr_acc.append(train_acc)

acc = torch.tensor(acc)
tr_acc = torch.tensor(tr_acc)
print_or_log(
    "---------------- Final Result ----------------",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
print_or_log(
    "Mean: {:7f}, Std: {:7f}".format(acc.mean(), acc.std()),
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
print_or_log(
    "Tr Mean: {:7f}, Std: {:7f}".format(tr_acc.mean(), tr_acc.std()),
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
print_or_log(
    "Average Acros Splits",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
print_or_log(
    "Training Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean_tr_accuracies = np.mean(tr_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean_tr_accuracies[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Testing Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean_tst_accuracies = np.mean(tst_accuracies, axis=1)
st_d_tst_accuracies = np.std(tst_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch "
        + str(epoch + 1)
        + ":"
        + str(mean_tst_accuracies[epoch])
        + "/"
        + str(st_d_tst_accuracies[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Testing Exp Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean_tst_e_accuracies = np.mean(tst_exp_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean_tst_e_accuracies[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Testing Lrn Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean_tst_l_accuracies = np.mean(tst_lrn_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean_tst_l_accuracies[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Perm Training Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean = np.mean(perm_tr_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Perm Test Acc:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean = np.mean(perm_tst_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Perm Training Loss:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean = np.mean(perm_tr_losses, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )

print_or_log(
    "Perm Test Loss:",
    log_file_path="log"
    + MODEL
    + DATASET
    + ","
    + str(LAYERS)
    + ","
    + str(WIDTH)
    + ".txt",
)
mean = np.mean(perm_tst_losses, axis=1)
for epoch in range(EPOCHS):
    print_or_log(
        "Epoch " + str(epoch + 1) + ":" + str(mean[epoch]),
        log_file_path="log"
        + MODEL
        + DATASET
        + ","
        + str(LAYERS)
        + ","
        + str(WIDTH)
        + ".txt",
    )
