import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

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

# Function to analyze and plot the effect of each hyperparameter
def analyze_hyperparameters(directory, datasets, models, default_values):
    """
    Analyzes and plots the effect of each hyperparameter on convergence for original and permuted datasets.
    """
    for dataset in datasets:
        for model in models:
            # Filter files for the current dataset and model
            pattern = f"{directory}/*{dataset}*{model.lower()}*.csv"
            files = glob.glob(pattern)

            for hyperparameter in ['DEPTH', 'WIDTH', 'NUM_RANDOM_FEATURES']:
                # 1st figure: Loss subplots
                plt.figure(figsize=(14, 10))
                plt.suptitle(
                    f'Effect of {hyperparameter} on Convergence\n{dataset} ({model.upper()}) - Loss',
                    fontsize=16
                )

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

                # Training Loss
                plt.subplot(2, 2, 1)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Train Loss'], label=f'{hyperparameter}={hyperparam_value} (Train Loss)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Epochs')
                plt.legend()
                plt.grid(True)

                # Permuted Training Loss
                plt.subplot(2, 2, 2)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Perm Train Loss'], label=f'{hyperparameter}={hyperparam_value} (Perm Train Loss)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Permuted Training Loss Over Epochs')
                plt.legend()
                plt.grid(True)

                # Permuted Test Loss
                plt.subplot(2, 2, 4)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Perm Test Loss'], label=f'{hyperparameter}={hyperparam_value} (Perm Test Loss)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Permuted Test Loss Over Epochs')
                plt.legend()
                plt.grid(True)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

                # 2nd figure: Accuracy subplots
                plt.figure(figsize=(14, 10))
                plt.suptitle(
                    f'Effect of {hyperparameter} on Convergence\n{dataset} ({model.upper()}) - Accuracy',
                    fontsize=16
                )

                # Training Accuracy
                plt.subplot(2, 2, 1)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Train Accuracy'], label=f'{hyperparameter}={hyperparam_value} (Train Accuracy)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Training Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)

                # Permuted Training Accuracy
                plt.subplot(2, 2, 2)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Perm Train Accuracy'], label=f'{hyperparameter}={hyperparam_value} (Perm Train Accuracy)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Permuted Training Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)

                # Permuted Test Accuracy
                plt.subplot(2, 2, 4)
                for file, depth, width, random_features in relevant_files:
                    df = pd.read_csv(file)
                    hyperparam_value = {
                        'DEPTH': depth,
                        'WIDTH': width,
                        'NUM_RANDOM_FEATURES': random_features
                    }[hyperparameter]
                    plt.plot(df['Epoch'], df['Perm Test Accuracy'], label=f'{hyperparameter}={hyperparam_value} (Perm Test Accuracy)', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Permuted Test Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

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
directory = 'true_results'

# Analyze the effect of hyperparameters
analyze_hyperparameters(directory, datasets, models, default_hyperparams)
