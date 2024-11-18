import torch
import torch_scatter
import torch.nn.functional as F
from torch_geometric.nn import GPSConv
from torch_geometric.nn import global_max_pool


class CustomGPS(torch.nn.Module):
    def __init__(self, model_hyperparams):
        super(CustomGPS, self).__init__()

        # Initialize the parent GAT class
        self.model = GPSConv(
            channels=model_hyperparams["channels"],
            conv=model_hyperparams.get(
                "conv", None
            ),  # If you need to define a custom MPNN layer
            heads=model_hyperparams.get("heads", 1),
            dropout=model_hyperparams.get("dropout", 0.0),
            act=model_hyperparams.get("act", "relu"),
            norm=model_hyperparams.get("norm", "batch_norm"),
            attn_type=model_hyperparams.get("attn_type", "multihead"),
            attn_kwargs=model_hyperparams.get(
                "attn_kwargs", None
            ),  # Additional normalization arguments
        )

        # print(
        #     model_hyperparams.get("channels", None),
        #     model_hyperparams.get("num_classes", None),
        # )

        self.fc = torch.nn.Linear(
            model_hyperparams.get("channels", None),
            model_hyperparams.get("num_classes", None),
        )

    def forward(self, x, edge_index, batch):
        # Forward pass through the GAT layers
        x = self.model.forward(x=x, edge_index=edge_index, batch=batch)

        # For graph-level tasks, apply global pooling
        x = global_max_pool(
            x, batch
        )  # or global_mean_pool(x, data.batch) for average pooling
        x = self.fc(x)  # Fully connected layer to match number of classes

        # Apply log softmax for final predictions
        return F.log_softmax(x, dim=1)


# Example usage for node-level prediction
# model = CustomGAT(in_channels=dataset.num_features, hidden_channels=64, out_channels=64, num_classes=dataset.num_classes, task='node')

# Example usage for graph-level prediction
# model = CustomGAT(in_channels=dataset.num_features, hidden_channels=64, out_channels=64, num_classes=dataset.num_classes, task='graph')
