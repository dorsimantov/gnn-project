import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GAT, GPSConv
from datasets import EXPDataset
import yaml


# Configuration file for training parameters and model settings
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type, model_params):
    if model_type == "GAT":
        return GAT(**model_params)
    elif model_type == "GPS":
        return GPSConv(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def test(model, data_loader, device):
    model.eval()
    correct = 0
    for data in data_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)


def main(config_path, data_path, use_new_data=False):
    # Load configuration
    config = load_config(config_path)

    # Dataset and DataLoader
    dataset = EXPDataset(data_path, use_new_data=use_new_data)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config['model']['type'], config['model']['params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['epochs']):
        train_loss = train(model, data_loader, optimizer, device)
        test_acc = test(model, data_loader, device)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main("config/train/baseline_EXP/GAT.yaml", "datasets/EXP", use_new_data=True)
