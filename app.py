import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Breast_Cancer.csv')

# 1. Encode the target variable 'Status' to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Status'])

# 2. Split data into features and encoded target
X = df.drop('Status', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Preprocessing for numerical and categorical columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# 4. Create a transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
            ('scaler', StandardScaler())  # Standardize numerical data
        ]), numerical_features),

        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with mode
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical data
        ]), categorical_features)
    ])

# 5. Define models to test
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# 6. Define parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'model__C': [0.1, 1, 10],
        'model__penalty': ['l2']
    },
    'Decision Tree': {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.3]
    }
}

# 7. Loop over each model, perform GridSearchCV and cross-validation
best_models = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Cross-validation results
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy for {model_name}: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")

    # Store the best model
    best_models[model_name] = grid_search.best_estimator_

# 8. Evaluate the best model on the test set
for model_name, best_model in best_models.items():
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy for {model_name}: {test_accuracy:.4f}")




# Initialize dictionary to store metrics for each model
metrics = {}

# Evaluate each model on the test set
for model_name, best_model in best_models.items():
    print(f"\nEvaluating {model_name}...")

    # Make predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    # Store metrics
    metrics[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# Visualization of metrics
plt.figure(figsize=(12, 8))

# Plotting the metrics for each model
sns.barplot(data=metrics_df.reset_index().melt(id_vars='index'),
            x='index', y='value', hue='variable')

# Formatting the plot
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Metrics')
plt.show()


