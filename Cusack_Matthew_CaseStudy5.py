## Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier  # Note: SGDClassifier for classification tasks
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import permutation_importance
import json
import numpy as np

## Load Data (replace 'your_data.csv' with your actual file path)
data = pd.read_csv('log2.csv')
print(data.shape)
print(data.columns)


## Separate features and target variable
X = data.drop('Action', axis=1)  # Replace 'target_column' with your actual class label column
y = data['Action']

## 10-Fold Training-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initiate SVM Model
svm_model = SVC()

## Initiate SGD Model (Note: Modified for classification)
sgd_model = SGDClassifier(loss='hinge')  # Hinge loss for classification with SGD

## Tune Model w/ GridSearchCV (replace param_grid with your desired parameters)
svm_param_grid = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}
sgd_param_grid = {'loss': ['hinge'], 'penalty': ['l1', 'l2'], 'alpha': [0.0001, 0.001, 0.01]}  # Adjust penalty and alpha

svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5, n_jobs=-1)
sgd_grid_search = GridSearchCV(sgd_model, sgd_param_grid, cv=5, n_jobs=-1)

## Train models
svm_grid_search.fit(X_train, y_train)
sgd_grid_search.fit(X_train, y_train)

## Save & Show Best Parameters
print("SVM Best Parameters:", svm_grid_search.best_params_)
print("SGD Best Parameters:", sgd_grid_search.best_params_)

# Access the best parameters
svm_best_params = svm_grid_search.best_params_
sgd_best_params = sgd_grid_search.best_params_

# Create a dictionary for combined storage
best_params = {
    'SVM': svm_best_params,
    'SGD': sgd_best_params
}

# Save to a JSON file
with open('best_params.json', 'w') as f:
    json.dump(best_params, f)

with open('best_params.json', 'r') as f:
    loaded_params = json.load(f)

# Access individual model's best parameters
best_svm_params = loaded_params['SVM']
best_sgd_params = loaded_params['SGD']

## Train models with best parameters (optional, only if needed)
best_svm_model = SVC(**svm_grid_search.best_params_)
best_sgd_model = SGDClassifier(**sgd_grid_search.best_params_)

best_svm_model.fit(X_train, y_train)
best_sgd_model.fit(X_train, y_train)

## Evaluate Models
svm_predictions = best_svm_model.predict(X_test)
sgd_predictions = best_sgd_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
sgd_accuracy = accuracy_score(y_test, sgd_predictions)

svm_report = classification_report(y_test, svm_predictions)
sgd_report = classification_report(y_test, sgd_predictions)

print("SVM Accuracy:", svm_accuracy)
print("SGD Accuracy:", sgd_accuracy)
print("SVM Classification Report:", svm_report)
print("SGD Classification Report:", sgd_report)

## Important Features (Further analysis)
svm_features = X_train.columns
svm_results = permutation_importance(best_svm_model, X_test, y_test, n_repeats=10)
svm_importances = svm_results.mean(axis=-1)
svm_importance_df = pd.DataFrame({'feature': svm_features, 'importance': svm_importances})
svm_importance_df = svm_importance_df.sort_values(by='importance', ascending=False)

# Print the most important features for SVM
print("\nMost important features for SVM:")
feature_importances = best_svm_model.feature_importances_
feature_names = X.columns

# Sort features by importance in descending order
sorted_idx = feature_importances.argsort()[::-1]
for idx, name in zip(sorted_idx, feature_names):
    print(f"{name}: {feature_importances[idx]:.4f}")

# Feature Importance for SGD (coefficients)
coefficients = best_sgd_model.coef_.flatten()  # Access model coefficients

sorted_idx = np.argsort(coefficients)[::-1]  # Sort indices for descending importance
feature_names = X.columns

for idx, name in zip(sorted_idx, feature_names):
    print(f"{name}: {coefficients[idx]:.4f}")