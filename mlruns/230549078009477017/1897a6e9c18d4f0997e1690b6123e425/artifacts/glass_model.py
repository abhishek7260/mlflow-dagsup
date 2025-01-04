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
import dagshub
# dagshub.init(repo_owner='abhishek7260', repo_name='mlflow-dagsup', mlflow=True)

mlflow.set_experiment("glass_model_gb")
# mlflow.set_tracking_uri("https://dagshub.com/abhishek7260/mlflow-dagsup.mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
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
# processed_df = remove_outlier_with_zscore(df)

# Split data into features (X) and target (y)
# x = processed_df.drop('Type', axis=1)
# y = processed_df['Type']

# Train-test split
train_data,test_data = train_test_split(df, test_size=0.2, random_state=42)

train_processed=remove_outlier_with_zscore(train_data)
test_processed=remove_outlier_with_zscore(test_data)
x_train=train_processed.drop('Type',axis=1)
y_train=train_processed['Type']
x_test=test_processed.drop('Type',axis=1)
y_test=test_processed['Type']
# Hyperparameter for Gradient Boosting
learning_rate = 0.01
n_estimators = 600
max_depth = 4

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
    
    train_df=mlflow.data.from_pandas(train_processed)
    test_df=mlflow.data.from_pandas(test_processed)
    
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
    
    
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")
    
    # Optional: Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))