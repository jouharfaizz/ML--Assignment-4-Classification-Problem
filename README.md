# ML--Assignment-4-Classification-Problem
Breast Cancer Classification Project
Overview

This project focuses on classifying breast cancer tumors as either malignant or benign using the breast cancer dataset from the sklearn library. Five supervised learning algorithms are implemented and compared:

    Logistic Regression

    Decision Tree Classifier

    Random Forest Classifier

    Support Vector Machine (SVM)

    k-Nearest Neighbors (k-NN)

The goal is to evaluate the performance of these algorithms and identify the best-performing model for this classification task.
Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, available in the sklearn.datasets module. It contains 569 samples with 30 features computed from digitized images of fine needle aspirates (FNA) of breast masses. The target variable is binary:

    0: Malignant

    1: Benign

Requirements

To run the code, you need the following Python libraries:

    scikit-learn

    pandas

    numpy

You can install the required libraries using the following command:
bash
Copy

pip install scikit-learn pandas numpy

Code Structure

The project is implemented in a Jupyter Notebook (breast_cancer_classification.ipynb). The notebook is organized into the following sections:

    Loading and Preprocessing:

        Load the dataset.

        Split the data into training and testing sets.

        Standardize the features using StandardScaler.

    Classification Algorithm Implementation:

        Implement five classification algorithms:

            Logistic Regression

            Decision Tree Classifier

            Random Forest Classifier

            Support Vector Machine (SVM)

            k-Nearest Neighbors (k-NN)

        Train and evaluate each model using accuracy as the metric.

    Model Comparison:

        Compare the performance of the five models.

        Identify the best and worst-performing models.

Results

The accuracy scores for the five models are as follows:
Algorithm	Accuracy
Logistic Regression	0.9766
Decision Tree	0.9415
Random Forest	0.9649
SVM	0.9766
k-NN	0.9649
Key Findings:

    Best Performing Model: Logistic Regression and SVM both achieved the highest accuracy of 97.66%.

    Worst Performing Model: Decision Tree had the lowest accuracy of 94.15%, likely due to overfitting.

How to Run the Code

    Clone this repository:
    bash
    Copy

    git clone https://github.com/your-username/breast-cancer-classification.git

    Navigate to the project directory:
    bash
    Copy

    cd breast-cancer-classification

    Open the Jupyter Notebook:
    bash
    Copy

    jupyter notebook breast_cancer_classification.ipynb

    Run the cells in the notebook to see the results.
