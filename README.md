# Online Shoppers Purchasing Intention — Classification

## Overview
This project uses machine learning to predict whether an online shopping session will result in a purchase (`Revenue`). It explores visitor-session data, compares classification algorithms, tunes a Random Forest model, and evaluates model performance.

## Project Objectives
- Inspect and preprocess the dataset.
- Explore purchase patterns across visitor types and months.
- Engineer a total page-duration feature.
- Compare multiple classification models using different train-test splits.
- Tune a Random Forest classifier with cross-validation.
- Evaluate the final model using classification metrics and a confusion matrix.
- Identify influential features using permutation importance.

## Dataset
The project uses the **Online Shoppers Purchasing Intention** dataset.

The target column is `Revenue`:
- `0` — no purchase
- `1` — purchase

The script expects the dataset CSV file to be available locally. Update the `DATA_PATH` variable in the Python script to match the location of your downloaded CSV file.

## Technologies and Libraries
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

## Machine Learning Models
The project compares the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Dummy Classifier baseline

The script evaluates models with 70–30, 75–25, and 80–20 train-test splits. It reports accuracy, precision, recall, and F1-score. A Random Forest model is then tuned using `GridSearchCV` and stratified 5-fold cross-validation.

## Workflow
1. Load the dataset and inspect its shape, missing values, and target distribution.
2. Check for duplicate rows and remove them.
3. Fill missing numeric values with the median and categorical values with the mode, when needed.
4. Convert Boolean columns to numeric values.
5. Visualize purchase distribution, visitor type, month, numeric correlations, and potential outliers.
6. Create `TotalPageDuration` from administrative, informational, and product-related page durations when those columns are available.
7. One-hot encode `Month` and `VisitorType`.
8. Compare classification models across the selected train-test splits.
9. Tune a Random Forest classifier with cross-validation.
10. Evaluate the tuned model and calculate permutation feature importance.

## Installation
Install the required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run
1. Download or place the dataset CSV on your computer.
2. Open the Python script.
3. Update `DATA_PATH` to the correct path for `online_shoppers_intention.csv`.
4. Run the script from your Python environment.

Example:

```bash
python online_shoppers_classification_complete_no_comments.py
```

## Outputs
When the script completes, it saves these files in the current working directory:
- `results.csv` — model comparison metrics for each train-test split.
- `feature_importance.csv` — permutation importance scores for the features.

It also displays charts for exploratory data analysis, model comparison, the confusion matrix, and feature importance.

## Evaluation Metrics
- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted purchases that are actual purchases.
- **Recall:** Proportion of actual purchases that the model correctly identifies.
- **F1-score:** Harmonic mean of precision and recall.

## Repository Structure

```text
.
├── online_shoppers_classification_complete_no_comments.py
├── README.md
└── online_shoppers_intention.csv   # Dataset file (add locally; may not be included in GitHub)
```

## Notes
- The script uses a local dataset path, so update `DATA_PATH` before running it on another computer.
- Model performance depends on the dataset and train-test split. Refer to the generated metrics rather than assuming a particular model is best.
- The dataset is not automatically downloaded by the script.

## Future Improvements
- Add a requirements file for reproducible environments.
- Save the trained model for reuse.
- Add cross-validation comparisons for all models.
- Build an interactive interface to enter session information and predict purchase intention.
