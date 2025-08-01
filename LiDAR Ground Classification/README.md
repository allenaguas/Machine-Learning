# LiDAR Ground Classification ML Project

This project demonstrates a simple machine learning workflow for classifying ground points in a LiDAR point cloud. The process involves data preparation, model training with a Random Forest classifier, and visualization of the results.

## Project Structure

```
LiDAR Ground Classification/
├── data/
│   ├── raw/
│   │   └── lassen_point_cloud.laz    # Input LiDAR data
│   └── processed/
│       ├── lassen_processed_data.npy # Processed features and labels
│       └── scaler.pkl                # Saved feature scaler
├── models/
│   └── ground_classifier_model.pkl   # Trained classification model
├── src/
│   ├── __init__.py
│   ├── data_preparation.py           # Script for data loading and preprocessing
│   ├── ground_classifier.py          # Script for model training and evaluation
│   └── utils.py                      # Utility functions (e.g., visualization)
├── run_simple_workflow.py            # Main script to run the entire workflow
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## How to Run

Follow these steps to set up the environment and run the complete workflow.

### 1. Setup

It is recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Run the Workflow

Execute the main script from the project's root directory. This script will automatically run the data preparation, model training, and visualization steps in sequence.

```bash
python3 run_simple_workflow.py
```

Upon completion, two Open3D windows will appear sequentially: one showing the ground truth classification and another showing the model's predictions.
