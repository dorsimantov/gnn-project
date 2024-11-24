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

# Function to analyze and save each subplot
def analyze_hyperparameters(directory, datasets, models, default_values):
    """
    Analyzes and saves individual subplots for each hyperparameter's effect on convergence.
    """
    for dataset in datasets:
        for model in models:
            # Filter files for the current dataset and model
            pattern = f"{directory}/logGNNHyb-tanh-0,0-n-True-lr0.00065-{dataset}*{model.lower()}*.csv"
            files = glob.glob(pattern)

            for hyperparameter in ['DEPTH', 'WIDTH', 'NUM_RANDOM_FEATURES']:
                # Filter files where only the current hyperparameter is varied
                relevant_files = []
                for file in files:
                    _, _, depth, width, model_name, random_features = parse_filename(file)
                    # Check if this file varies only the current hyperparameter
                    is_relevant = (
                        (hyperparameter == 'DEPTH' and width == default_values['WIDTH'] and random_features == default_values['NUM_RANDOM_FEATURES']) or
                        (hyperparameter == 'WIDTH' and depth == default_values['DEPTH'] and random_features == default_values['NUM_RANDOM_FEATURES']) or
                        (hyperparameter == 'NUM_RANDOM_FEATURES' and depth == default_values['DEPTH'] and width == default_values['WIDTH'])
                    )
                    if is_relevant:
                        relevant_files.append((file, depth, width, random_features))

                # Save Loss subplots
                for idx, (data, title, ylabel, filename_suffix) in enumerate([
                    ('Train Loss', 'Training Loss Over Epochs', 'Loss', 'train_loss'),
                    ('Perm Train Loss', 'Permuted Training Loss Over Epochs', 'Loss', 'perm_train_loss'),
                    ('Perm Test Loss', 'Permuted Test Loss Over Epochs', 'Loss', 'perm_test_loss')
                ]):
                    plt.figure(figsize=(7, 5))
                    for file, depth, width, random_features in relevant_files:
                        df = pd.read_csv(file)
                        hyperparam_value = {
                            'DEPTH': depth,
                            'WIDTH': width,
                            'NUM_RANDOM_FEATURES': random_features
                        }[hyperparameter]
                        plt.plot(df['Epoch'], df[data], label=f'{hyperparameter}={hyperparam_value}', alpha=0.7)
                    plt.xlabel('Epoch', fontsize=20)
                    plt.ylabel(ylabel, fontsize=20)
                    plt.xticks(fontsize=12)  # Adjust the font size for x-ticks
                    plt.yticks(fontsize=12)  # Adjust the font size for y-ticks
                    plt.legend(fontsize=18)
                    plt.grid(True)

                    # Save the subplot to an SVG file
                    Path('figures').mkdir(parents=True, exist_ok=True)
                    svg_filename = f"figures/{dataset}_{model}_{hyperparameter}_{filename_suffix}.svg"
                    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
                    plt.close()

                # Save Accuracy subplots
                for idx, (data, title, ylabel, filename_suffix) in enumerate([
                    ('Train Accuracy', 'Training Accuracy Over Epochs', 'Accuracy', 'train_accuracy'),
                    ('Test Accuracy', 'Test Accuracy Over Epochs', 'Accuracy', 'test_accuracy'),
                    ('Perm Train Accuracy', 'Permuted Training Accuracy Over Epochs', 'Accuracy', 'perm_train_accuracy'),
                    ('Perm Test Accuracy', 'Permuted Test Accuracy Over Epochs', 'Accuracy', 'perm_test_accuracy')
                ]):
                    plt.figure(figsize=(7, 5))
                    for file, depth, width, random_features in relevant_files:
                        df = pd.read_csv(file)
                        hyperparam_value = {
                            'DEPTH': depth,
                            'WIDTH': width,
                            'NUM_RANDOM_FEATURES': random_features
                        }[hyperparameter]
                        plt.plot(df['Epoch'], df[data], label=f'{hyperparameter}={hyperparam_value}', alpha=0.7)
                    plt.xlabel('Epoch', fontsize=20)
                    plt.ylabel(ylabel, fontsize=20)
                    plt.xticks(fontsize=12)  # Adjust the font size for x-ticks
                    plt.yticks(fontsize=12)  # Adjust the font size for y-ticks
                    plt.legend(fontsize=18)
                    plt.grid(True)

                    # Save the subplot to an SVG file
                    Path('figures').mkdir(parents=True, exist_ok=True)
                    svg_filename = f"figures/{dataset}_{model}_{hyperparameter}_{filename_suffix}.svg"
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

# Analyze the effect of hyperparameters
analyze_hyperparameters(directory, datasets, models, default_hyperparams)
