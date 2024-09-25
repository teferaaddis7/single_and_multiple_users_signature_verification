import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

# Load the dataset
data_path = '/home/asset/Desktop/DTW_New/Online_signature_snadeepDataset/dataset/HnS_AllFeaturesWithHeader.csv'
df = pd.read_csv(data_path, delimiter=',')
print("Dataset loaded successfully:")
print(df.head())

# Label the data
n_users = 30
samples_per_user = 60
labels = np.repeat(np.arange(n_users), samples_per_user)
df['label'] = labels

# Prepare the data for one-class verification
X = df.iloc[:, :-1].values
y = df['label'].values

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Initialize results storage
results = []

# Perform cross-validation for each user
for target_user in range(n_users):
    # Extract samples of the target user for training
    X_train_target = X[y == target_user]

    # Initialize and train the One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    oc_svm.fit(X_train_target)

    # Test against all samples
    y_pred = oc_svm.predict(X)
    
    # Convert One-Class SVM output to binary labels
    y_pred_binary = np.where(y_pred == 1, target_user, -1)
    
    # Evaluate the model for the current target user
    accuracy = accuracy_score(y == target_user, y_pred_binary == target_user)
    roc_auc = roc_auc_score(y == target_user, y_pred)
    cm = confusion_matrix(y == target_user, y_pred_binary == target_user)
    class_report = classification_report(y == target_user, y_pred_binary == target_user, output_dict=True)
    
    results.append({
        "target_user": target_user,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": class_report
    })

    print(f"User {target_user}: One-Class SVM Accuracy: {accuracy*100:.2f}%")
    print(f"User {target_user}: One-Class SVM ROC AUC: {roc_auc:.2f}")
    print(f"User {target_user}: One-Class SVM Classification Report:")
    print(pd.DataFrame(class_report).transpose())
    print(f"User {target_user}: One-Class SVM Confusion Matrix:")
    plot_confusion_matrix(cm, classes=['Non-Target', 'Target'])

# Print summary results
for result in results:
    print(f"User {result['target_user']}: One-Class SVM Accuracy: {result['accuracy']*100:.2f}%")
    print(f"User {result['target_user']}: One-Class SVM ROC AUC: {result['roc_auc']:.2f}")
    print(pd.DataFrame(result['classification_report']).transpose())
    print(result['confusion_matrix'])
