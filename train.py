# TODO: Get current training task from base.yaml
# TODO: Read task hyperparams from config/train/TRAIN_NAME/train_info.yaml into a dictionary
# TODO: Read model hyperparams (for specific task) from config/train/TRAIN_NAME/MODEL_NAME.yaml into a dictionary
# TODO: Train the model
#  Dump training weights in models/weights/TRAIN_NAME, unless we disable weights saving (in either base config, task info or model config for task)
#  Dump training logs in models/weights/TRAIN_NAME/logs
# TODO: Save task hyperparams from config/train/TRAIN_NAME/train_info.yaml and current time as train_info.json in both results/train/TRAIN_NAME and models/MODEL_NAME/weights/TRAIN_NAME
# TODO: Save model hyperparams from config/train/TRAIN_NAME/MODEL_NAME.yaml and current time as MODEL_NAME.json in both results/train/TRAIN_NAME/MODEL_NAME and models/MODEL_NAME/weights/TRAIN_NAME
# TODO: Dump results in the results/train/TRAIN_NAME/MODEL_NAME (separate folders for outputs, plots, and metrics)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.loader import DataLoader
import yaml

# Load parameters from YAML file
def load_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Training loop
def train(model, loader, optimizer, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

# Evaluation function
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    return correct / sum([data.test_mask.sum().item() for data in loader])

# Main function
def main(config_path):
    # Load config
    config = load_config(config_path)
    model_params = config['model']
    training_params = config['training']

    # Initialize model and optimizer
    device = torch.device(training_params['device'])
    model = GAT(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])

    # Assuming `train_loader` and `test_loader` are predefined DataLoader objects
    # Replace with actual dataset and DataLoader setup as needed
    train_loader = DataLoader(...)
    test_loader = DataLoader(...)

    # Training loop
    for epoch in range(training_params['epochs']):
        train(model, train_loader, optimizer, device)
        acc = test(model, test_loader, device)
        print(f'Epoch {epoch + 1}, Accuracy: {acc:.4f}')

# Run the main function
if __name__ == "__main__":
    main("config/train/baseline/GAT.yaml")
