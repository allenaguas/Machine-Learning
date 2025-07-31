# Project Explanation: LiDAR Ground Classification

This document provides a detailed breakdown of the machine learning workflow used in this project. The goal is to classify points in a LiDAR point cloud as either "Ground" or "Non-Ground".

## The Overall Workflow

The entire process is managed by the `run_simple_workflow.py` script. It executes three main stages in sequence:

1.  **Data Preparation**: Loads the raw LiDAR data, extracts useful features, and prepares it for the model.
2.  **Model Training**: Trains a machine learning model to distinguish between ground and non-ground points.
3.  **Visualization**: Shows the results by comparing the original data's labels with the model's predictions.

---

## Stage 1: Data Preparation (`src/data_preparation.py`)

This stage is responsible for turning the raw point cloud into a clean, numerical format that a machine learning model can understand.

### Key Steps:

1.  **Load LAZ File**:
    *   It reads the `data/raw/lassen_point_cloud.laz` file. A `.laz` file is a compressed version of a `.las` file, which is a standard format for LiDAR data.
    *   The `laspy` library, along with the `lazrs` backend, is used to handle the decompression and reading of the file.

2.  **Extract Point Data**:
    *   **Coordinates (Features)**: It extracts the X, Y, and Z coordinates for every point in the cloud. These three values are the initial *features* our model will use to learn.
    *   **Classification (Labels)**: It also extracts the pre-existing `classification` value for each point. In the LAS format, points are often already labeled (e.g., class `2` is Ground). This is our "ground truth" data, which we will use to train and evaluate the model.

3.  **Feature Scaling**:
    *   Machine learning models, including Random Forest, often perform better when the input features are on a similar scale. The X, Y, and Z coordinates can have very different ranges.
    *   `StandardScaler` from `scikit-learn` is used to transform the features. It rescales the data to have a mean of 0 and a standard deviation of 1.
    *   The scaler object itself is saved to `data/processed/scaler.pkl`. This is crucial because if we ever want to use our trained model to predict on *new* data, we must scale that new data in the exact same way.

4.  **Save Processed Data**:
    *   All the prepared data (the original XYZ points for visualization, the scaled features for the model, and the ground truth labels) is saved into a single NumPy file: `data/processed/lassen_processed_data.npy`. This makes it fast and easy to load in the next stage.

---

## Stage 2: Model Training & Evaluation (`src/ground_classifier.py`)

This stage takes the prepared data and uses it to train and test our classification model.

### Key Steps:

1.  **Load Processed Data**:
    *   It loads the `lassen_processed_data.npy` file created in the previous stage.

2.  **Split the Data**:
    *   The data is split into two sets: a **training set** (80% of the data) and a **testing set** (20%).
    *   The model will only ever "see" the training set during the learning process. The testing set is held back and used to evaluate how well the model performs on unseen data.
    *   We use `stratify=labels` to ensure that the proportion of ground vs. non-ground points is the same in both the training and testing sets. This is important for getting a reliable evaluation.

3.  **Train the Model**:
    *   We use a `RandomForestClassifier`. A Random Forest is an "ensemble" model, meaning it's built from many individual decision trees. It's a powerful and popular choice for classification because it's robust and less prone to overfitting than a single decision tree.
    *   The model is trained by calling `model.fit(X_train, y_train)`. During this step, the algorithm looks for patterns in the scaled XYZ coordinates that correspond to the ground truth labels.

4.  **Evaluate Performance**:
    *   The trained model is asked to predict the labels for the testing set (`model.predict(X_test)`).
    *   The predictions are compared to the actual labels (`y_test`).
    *   A **Classification Report** is printed, showing metrics like `precision`, `recall`, and `f1-score` for each class. This gives a detailed view of the model's performance.
    *   The overall **Accuracy** (the percentage of correctly predicted labels) is also calculated and printed.

5.  **Save the Trained Model**:
    *   The final, trained model object is saved to `models/ground_classifier_model.pkl` using `joblib`. This file contains the complete, ready-to-use classifier.

---

## Stage 3: Visualization (`run_simple_workflow.py` & `src/utils.py`)

The final stage allows us to visually inspect the model's performance.

### Key Steps:

1.  **Load Data and Model**:
    *   The main workflow script loads the original (unscaled) points from the `.npy` file and the trained model from the `.pkl` file.

2.  **Create a Color Map**:
    *   A simple color scheme is defined to make the visualization clear:
        *   **Ground** points (label 2) will be colored **dark grey**.
        *   **Non-Ground** points (label 1) will be colored **red**.

3.  **Visualize Ground Truth**:
    *   The first visualization window shows the point cloud colored according to its original, ground-truth labels. This is what the "perfect" classification looks like.

4.  **Visualize Model Predictions**:
    *   The script then uses the trained model to predict the class for *every point* in the dataset.
    *   A second visualization window opens, showing the point cloud colored according to the model's predictions.

By comparing the two windows, you can visually assess where the model performed well and where it made mistakes.
