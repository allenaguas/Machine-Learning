# run_simple_workflow.py
# Author: Allen Aguas
# Description: Orchestrates the LiDAR ground classification workflow.

"""
Project Workflow Orchestrator

This is the main script to run the entire project. It executes the necessary
steps in sequence:
1. Prepares the data by calling the data preparation module.
2. Trains the model by calling the classifier module.
3. Loads the results and visualizes the ground truth and model predictions.
"""

import os
import sys
import numpy as np
import joblib
import argparse
import shutil

# --- 1. Define Project Root and Add to Python Path ---
# This makes all file paths robust and allows for consistent module imports.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src import data_preparation, ground_classifier, utils
except ImportError as e:
    print(f"Error: Could not import project modules from 'src' directory: {e}")
    print("Please ensure the 'src' directory and its files exist.")
    """Removes generated files from the 'processed' and 'models' directories."""
    print("\n--- Cleaning Project ---")
    
    processed_dir = os.path.join(root_path, "data/processed")
    models_dir = os.path.join(root_path, "models")
    
    dirs_to_clean = [processed_dir, models_dir]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing directory: {d}")
            shutil.rmtree(d)
        else:
            print(f"Directory does not exist, skipping: {d}")
    
    print("--- Cleaning complete. ---\n")


def visualize_results(root_path):
    """Loads data and model to visualize ground truth vs. predictions."""
    print("\n--- Running Visualization ---")
    try:
        npy_file_path = os.path.join(root_path, "data/processed/lassen_processed_data.npy")
        model_file_path = os.path.join(root_path, "models/ground_classifier_model.pkl")

        if not (os.path.exists(npy_file_path) and os.path.exists(model_file_path)):
            print("Skipping visualization: Could not find required files.")
            if not os.path.exists(npy_file_path):
                print(f"  - Missing: {npy_file_path}")
            if not os.path.exists(model_file_path):
                print(f"  - Missing: {model_file_path}")
            return

        print(f"Loading data from {npy_file_path}...")
        data = np.load(npy_file_path, allow_pickle=True).item()
        original_points = data['original_points']
        scaled_features = data['scaled_features']
        ground_truth_labels = data['labels']

        # --- Create a color map for visualization ---
        # We create a map that is large enough to hold all unique label values.
        # According to the LAS specification, '2' is Ground and '1' is often Unclassified (Non-Ground).
        unique_labels = np.unique(ground_truth_labels)
        max_label = np.max(unique_labels) if len(unique_labels) > 0 else 0
        # Default color is a light gray for any other classes.
        color_map = np.full((max_label + 1, 3), [0.8, 0.8, 0.8])

        # Assign specific colors for known labels if they exist in the data
        if 1 <= max_label:
            color_map[1] = [1.0, 0.0, 0.0]  # Non-Ground -> Red
        if 2 <= max_label:
            color_map[2] = [0.5, 0.5, 0.5]  # Ground -> Dark Grey

        print("Visualizing Ground Truth Labels...")
        print(">>> Please close the visualization window to continue. <<<")
        utils.visualize_classification_results(
            original_points, ground_truth_labels, "Ground Truth Labels", color_map
        )

        print("Loading trained model for prediction visualization...")
        model = joblib.load(model_file_path)
        predicted_labels = model.predict(scaled_features)

        print("Visualizing Model Predictions...")
        print(">>> Please close the visualization window to complete the workflow. <<<")
        utils.visualize_classification_results(
            original_points, predicted_labels, "Random Forest Predictions", color_map
        )

    except Exception as e:
        print(f"An error occurred during visualization: {e}")


def main():
    """Main function to orchestrate the workflow."""
    # --- Handle Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Run the LiDAR ML workflow.")
    parser.add_argument('--clean', action='store_true', help="Clean the project by removing generated files before running.")
    args = parser.parse_args()

    if args.clean:
        clean_project(PROJECT_ROOT)

    # --- Run data_preparation.py ---
    try:
        data_preparation.main(project_root=PROJECT_ROOT)
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        sys.exit(1)

    # --- Run ground_classifier.py ---
    try:
        ground_classifier.main(project_root=PROJECT_ROOT)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        sys.exit(1)

    # --- Visualize Results ---
    visualize_results(PROJECT_ROOT)

    # --- Workflow Completion ---
    print("\n--- Full project workflow completed ---")


if __name__ == "__main__":
    main()