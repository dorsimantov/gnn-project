import os
import time
import yaml
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GPSConv
from torch_geometric.nn.models import GAT
from torch_geometric.loader import DataLoader
import datasets
import pprint

from models.GAT.model import CustomGAT


# Helper function to load a YAML file
def load_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


# Load base configuration to get current task
base_config = load_yaml("config/base.yaml")
train_name = base_config["task"]["train_task"]
model_name = base_config["task"]["model"]

# 1. Read task hyperparameters from config/train/TRAIN_NAME/train_info.yaml
# TODO: any training hyperparams shared between models at all?
# For now we do nothing with these (yaml is empty)
task_config_path = f"config/train/{train_name}/train_info.yaml"
task_hyperparams = load_yaml(task_config_path)
dataset_name = task_hyperparams["dataset"]["dataset_name"]

# 2. Read model-specific hyperparameters for the selected task
task_for_model_config_path = f"config/train/{train_name}/{model_name}.yaml"
task_for_model_hyperparams = load_yaml(task_for_model_config_path)
train_hyperparams = task_for_model_hyperparams["training"]
model_hyperparams = task_for_model_hyperparams["model"]

# 3. Set up directories for saving logs, weights, and results
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"models/{model_name}/weights/{train_name}/{current_time}"
os.makedirs(save_dir, exist_ok=True)

logs_dir = os.path.join(save_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# 4. Setup TensorBoard for logging (optional)
writer = SummaryWriter(log_dir=logs_dir)

# 5. Choose the model (GPSConv or custom GAT from models.GAT)
device = torch.device(train_hyperparams["device"])

if model_name == "GPS":
    # GPSConv model (custom graph transformer layer)
    model = GPSConv(
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
        ),  # Additional attention args
    ).to(device)

elif model_name == "GAT":
    # Use the custom GAT model defined in models.GAT
    model = CustomGAT(
        model_hyperparams,
    ).to(device)

else:
    raise ValueError(f"Unsupported model: {model_name}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=train_hyperparams["learning_rate"],
    weight_decay=train_hyperparams["weight_decay"],
)

# Define the dataset and a DataLoader
if dataset_name == "EXP":
    dataset = datasets.EXPDataset().to(device)
elif dataset_name == "CEXP":
    dataset = datasets.CEXPDataset().to(device)
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

train_loader = DataLoader(
    dataset, batch_size=train_hyperparams["batch_size"], shuffle=True
)

train_losses = []
train_accuracies = []

# Training loop (simplified)
for epoch in range(train_hyperparams["epochs"]):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    # pprint.pp(train_loader.dataset)
    for batch in train_loader:
        print(batch.x)
        print(f"x dim: {batch.x.shape}")
        print(batch.y)
        print(f"y dim: {batch.y.shape}")
        print(batch.edge_index)
        print(f"edge index shape: {batch.edge_index.shape}")
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index)
        print(f"output dim: {output.shape}")
        loss = F.nll_loss(output, batch.y)  # Adjust with actual loss calculation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Compute accuracy (assuming binary classification for simplicity)
        predicted_labels = output.argmax(dim=1)
        correct_predictions += (predicted_labels == batch.y).sum().item()
        total_samples += batch.y.size(0)

    # Log to TensorBoard
    writer.add_scalar("Loss/train", total_loss, epoch)

    # Calculate and log accuracy
    accuracy = correct_predictions / total_samples
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    # Append to list for later plotting
    train_losses.append(total_loss)
    train_accuracies.append(accuracy)

    # Save weights at specified intervals or after training
    if epoch % 10 == 0 or epoch == model_hyperparams["training"]["epochs"] - 1:
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"weights_epoch_{epoch}.pth")
        )

# 6. Save task and train hyperparameters and current time as train_info.json
train_info = {
    "task_hyperparams": task_hyperparams,
    "train_hyperparams": train_hyperparams,
    "current_time": current_time,
}
train_info_path = os.path.join(save_dir, "train_info.json")
with open(train_info_path, "w") as f:
    json.dump(train_info, f, indent=4)

# 7. Save model hyperparameters and current time as MODEL_NAME.json
model_info = {"model_hyperparams": model_hyperparams, "current_time": current_time}
model_info_path = os.path.join(save_dir, f"{model_name}.json")
with open(model_info_path, "w") as f:
    json.dump(model_info, f, indent=4)

# 8. Dump results in results/train/TRAIN_NAME/MODEL_NAME
results_dir = f"results/train/{train_name}/{model_name}/{current_time}"
os.makedirs(results_dir, exist_ok=True)

# Save any output or metrics (example)
output_path = os.path.join(results_dir, "output.txt")
with open(output_path, "w") as f:
    f.write("Training completed successfully.\n")

# Example of saving plots (loss and accuracy)
# Plot and save loss
plt.figure(figsize=(8, 6))
plt.plot(range(train_hyperparams["epochs"]), train_losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
loss_plot_path = os.path.join(results_dir, "training_loss.png")
plt.savefig(loss_plot_path)
plt.close()

# Plot and save accuracy
plt.figure(figsize=(8, 6))
plt.plot(
    range(train_hyperparams["epochs"]),
    train_accuracies,
    label="Accuracy",
    color="green",
)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.legend()
accuracy_plot_path = os.path.join(results_dir, "training_accuracy.png")
plt.savefig(accuracy_plot_path)
plt.close()

# Close TensorBoard writer
writer.close()
