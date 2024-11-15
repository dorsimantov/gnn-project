import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from datasets import EXPDataset

# Load a graph dataset (e.g., MUTAG)
# dataset = TUDataset(root="data/TUDataset", name="MUTAG")
dataset = EXPDataset().to("cuda")

# Split dataset into train and test sets
torch.manual_seed(42)
train_dataset = dataset[: 3 * len(dataset) // 4]
test_dataset = dataset[3 * len(dataset) // 4 :]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        # First GAT layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        # Second GAT layer
        self.conv2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=0.3,
        )
        # Fully connected layer for graph classification
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Apply first GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Apply second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        # Perform graph-level pooling
        x = global_mean_pool(x, batch)
        # Classification logits
        x = self.fc(x)
        # Log-probabilities for NLLLoss
        return F.log_softmax(x, dim=1)


# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAT(
    input_dim=dataset.num_node_features,
    hidden_dim=64,
    output_dim=dataset.num_classes,
    num_heads=4,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)


# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)  # Negative Log-Likelihood Loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Testing function
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Get predicted class
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


# Training loop
for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(
        f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
