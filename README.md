# Advanced K-Nearest Neighbors (KNN) Classification

## Overview
This project implements an advanced **K-Nearest Neighbors (KNN)** machine learning algorithm using Python and Scikit-learn. It demonstrates classification, feature scaling, hyperparameter tuning, cross-validation, and model evaluation using a built-in breast cancer dataset.

## Aim
To develop an advanced KNN classification model that automatically selects suitable hyperparameters and evaluates prediction performance.

## Features
- Object-oriented implementation using a Python class.
- Feature scaling using `StandardScaler`.
- Automatic hyperparameter tuning using `GridSearchCV`.
- Five-fold cross-validation.
- Comparison of Euclidean and Manhattan distance metrics.
- Uniform and distance-based neighbor weighting.
- Model evaluation using accuracy, precision, recall, and F1-score.
- Prediction on unseen test samples.

## Technologies Used
- Python 3
- NumPy
- Scikit-learn
- Git and GitHub

## Project Structure
```text
cicd-exp-01-git-operations/
├── advanced_knn.py
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Nisanth-25-Tech/cicd-exp-01-git-operations.git
   ```

2. Navigate to the project directory:

   ```bash
   cd cicd-exp-01-git-operations
   ```

3. Install the dependencies:

   ```bash
   pip install numpy scikit-learn
   ```

## Execution

Run the Python program:

```bash
python advanced_knn.py
```

The program loads the dataset, splits it into training and testing sets, searches for the best KNN parameters, evaluates the selected model, and predicts a test sample's class.

## Algorithm Workflow
1. Load the breast cancer dataset.
2. Split the dataset into training and testing sets.
3. Scale features using a machine learning pipeline.
4. Tune the number of neighbors, distance metric, and weighting method.
5. Train the best-performing KNN configuration.
6. Evaluate the model using test data.
7. Display classification metrics and a sample prediction.

## Expected Output
The program displays:
- Best hyperparameters.
- Best cross-validation accuracy.
- Test-set accuracy.
- Classification report.
- Predicted class for a test sample.

*Actual results depend on the dataset, library versions, and train-test split.*

## Learning Outcomes
- Understand the KNN classification algorithm.
- Learn feature scaling and distance-based classification.
- Apply cross-validation and hyperparameter optimization.
- Evaluate a machine learning model using classification metrics.
- Organize and maintain Python projects using Git and GitHub.

## Conclusion
The project demonstrates an advanced KNN classification workflow using Scikit-learn. Hyperparameter tuning and feature scaling help build a more systematic classification model, while cross-validation and evaluation metrics provide a way to assess its performance.

**Note:** This educational project uses a public breast cancer dataset for machine learning demonstrations. It is not intended for medical diagnosis.
