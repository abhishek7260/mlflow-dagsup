import pandas as pd
import numpy as np
from scipy.stats import zscore
import pickle
import yaml
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn

mlflow.set_experiment("glass_model_gb")
mlflow.set_tracking_uri("http://localhost:5000")
# Load the dataset
df = pd.read_csv("E:\\glass_prediction_mlflow\\data\\glass.csv")
df.head()

# Function to remove outliers based on Z-score
def remove_outlier_with_zscore(df, columns=None, threshold=3):
    """
    Removes outliers from the dataframe based on Z-score.

    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - columns (list or None): List of columns to process. If None, numeric columns are considered.
    - threshold (float): The Z-score threshold to determine outliers (default is 3).

    Returns:
    - pd.DataFrame: Dataframe with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in columns:
        # Calculate Z-scores
        df['zscore'] = zscore(df[col])
        
        # Filter rows where the absolute Z-score is below the threshold
        df = df[abs(df['zscore']) <= threshold]
        
        # Drop the temporary Z-score column
        df = df.drop(columns=['zscore'])
    
    return df 

# Remove outliers
processed_df = remove_outlier_with_zscore(df)

# Split data into features (X) and target (y)
x = processed_df.drop('Type', axis=1)
y = processed_df['Type']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Hyperparameter for Gradient Boosting
learning_rate = 0.1
n_estimators = 500
max_depth = 3

# Training with MLflow logging
with mlflow.start_run():
    # Initialize and train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)
    
    # Save the model
    pickle.dump(model, open("gradient_boosting_model.pkl", "wb"))
    
    # Load the model and make predictions
    model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
    y_pred = model.predict(x_test)
    
    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
    recall = recall_score(y_test, y_pred, average='macro')
    cm=confusion_matrix(y_test,y_pred)
    # plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")  
    plt.ylabel("Actual")
    plt.title("confusion matrix") 
    plt.savefig("confusion_matrix.png") 
    mlflow.log_artifact("confusion_matrix.png")
    # Log metrics and parameters to MLflow
    mlflow.sklearn.log_model(model,"GradientBoostingClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_artifact(__file__)
    
    # Optional: Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))