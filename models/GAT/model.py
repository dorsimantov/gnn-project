import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_max_pool


class CustomGAT(torch.nn.Module):
    def __init__(self, model_hyperparams):
        super(CustomGAT, self).__init__()

        # Initialize the parent GAT class
        self.model = GAT(
            in_channels=model_hyperparams["in_channels"],  # Input feature size
            hidden_channels=model_hyperparams["hidden_channels"],  # Hidden layer size
            num_layers=model_hyperparams[
                "num_layers"
            ],  # Number of message passing layers
            out_channels=model_hyperparams.get(
                "out_channels", None
            ),  # Output size (optional)
            v2=model_hyperparams.get("v2", False),  # GATv2 flag
            dropout=model_hyperparams.get("dropout", 0.0),  # Dropout probability
            act=model_hyperparams.get("act", "relu"),  # Activation function
            act_first=model_hyperparams.get(
                "act_first", False
            ),  # Apply activation before normalization
            norm=model_hyperparams.get("norm", None),  # Normalization function
            jk=model_hyperparams.get("jk", None),  # Jumping Knowledge mode
            act_kwargs=model_hyperparams.get(
                "act_kwargs", None
            ),  # Additional activation arguments
            norm_kwargs=model_hyperparams.get(
                "norm_kwargs", None
            ),  # Additional normalization arguments
        )

        print(
            model_hyperparams.get("out_channels", None),
            model_hyperparams.get("num_classes", None),
        )

        self.fc = torch.nn.Linear(
            model_hyperparams.get("out_channels", None),
            model_hyperparams.get("num_classes", None),
        )

    def forward(self, x, edge_index):
        # Forward pass through the GAT layers
        x = self.model.forward(x, edge_index)

        # For graph-level tasks, apply global pooling
        x = global_max_pool(
            x, data.batch
        )  # or global_mean_pool(x, data.batch) for average pooling
        x = self.fc(x)  # Fully connected layer to match number of classes

        # Apply log softmax for final predictions
        return F.log_softmax(x, dim=1)


# Example usage for node-level prediction
# model = CustomGAT(in_channels=dataset.num_features, hidden_channels=64, out_channels=64, num_classes=dataset.num_classes, task='node')

# Example usage for graph-level prediction
# model = CustomGAT(in_channels=dataset.num_features, hidden_channels=64, out_channels=64, num_classes=dataset.num_classes, task='graph')
