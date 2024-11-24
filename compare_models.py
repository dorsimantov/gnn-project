import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
from pathlib import Path

# Function to parse filename and extract details
def parse_filename(filename):
    """
    Parses the filename to extract dataset, depth, width, model name, and number of random features.
    """
    lr_match = re.search(r"lr([\d.]+)", filename)
    learning_rate = lr_match.group(1) if lr_match else "Unknown"

    pattern = r"lr[\d.]+-([\w]+),(\d+),(\d+),([\w]+),(\d+)"
    match = re.search(pattern, filename)
    if match:
        dataset, depth, width, model_name, random_features = match.groups()
        model_name = model_name.upper()  # Format as uppercase (e.g., GINConv)
        return learning_rate, dataset, int(depth), int(width), model_name, int(random_features)
    return learning_rate, "Unknown", None, None, "Unknown", None

# Function to create comparison graphs for default hyperparameters
def compare_models(directory, datasets, models, default_values):
    """
    Creates comparison graphs for different models on each dataset using default hyperparameters.
    """
    metrics = [
        ('Train Loss', 'Training Loss Over Epochs', 'Loss', 'train_loss'),
        ('Perm Train Loss', 'Permuted Training Loss Over Epochs', 'Loss', 'perm_train_loss'),
        ('Perm Test Loss', 'Permuted Test Loss Over Epochs', 'Loss', 'perm_test_loss'),
        ('Train Accuracy', 'Training Accuracy Over Epochs', 'Accuracy', 'train_accuracy'),
        ('Perm Train Accuracy', 'Permuted Training Accuracy Over Epochs', 'Accuracy', 'perm_train_accuracy'),
        ('Test Accuracy', 'Test Accuracy Over Epochs', 'Accuracy', 'test_accuracy'),
        ('Perm Test Accuracy', 'Permuted Test Accuracy Over Epochs', 'Accuracy', 'perm_test_accuracy'),
    ]

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for metric, title, ylabel, filename_suffix in metrics:
            plt.figure(figsize=(10, 6))
            for model in models:
                # Filter files for the current dataset and model with default hyperparameters
                pattern = (
                    f"{directory}/logGNNHyb-tanh-0,0-n-True-lr0.00065-{dataset},"
                    f"{default_values['DEPTH']},{default_values['WIDTH']},{model},{default_values['NUM_RANDOM_FEATURES']}.csv"
                )
                files = glob.glob(pattern)
                if not files:
                    continue

                for file in files:
                    df = pd.read_csv(file)
                    plt.plot(df['Epoch'], df[metric], label=model.upper() if model.upper() != 'DEFAULT' else 'GRAPHCONV', alpha=0.7)

            # Plot settings
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.xticks(fontsize=12)  # Adjust the font size for x-ticks
            plt.yticks(fontsize=12)  # Adjust the font size for y-ticks
            plt.legend(fontsize=18)
            plt.grid(True)

            # Save the graph to an SVG file
            Path('figures').mkdir(parents=True, exist_ok=True)
            svg_filename = f"figures/{dataset}_comparison_{filename_suffix}.svg"
            plt.savefig(svg_filename, format='svg', bbox_inches='tight')
            plt.close()

# Default hyperparameter values
default_hyperparams = {
    'DEPTH': 8,
    'WIDTH': 64,
    'NUM_RANDOM_FEATURES': 1
}

# List of datasets and models
datasets = ['EXP', 'CEXP', 'NCI1']
models = ['ginconv', 'sageconv', 'default']

# Directory containing CSV files
directory = 'results'

# Compare models for default hyperparameters
compare_models(directory, datasets, models, default_hyperparams)
