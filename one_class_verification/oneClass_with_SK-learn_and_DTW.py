import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

# Function to compute DTW distance matrix
def compute_dtw_distance_matrix(X_train, X_test):
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    train_distance_matrix = np.zeros((n_train_samples, n_train_samples))
    test_distance_matrix = np.zeros((n_test_samples, n_train_samples))
    
    print("Computing DTW distance matrix for training data...")
    for i in range(n_train_samples):
        if i % 10 == 0:
            print(f"Training sample {i}/{n_train_samples}")
        for j in range(i, n_train_samples):
            distance, _ = fastdtw(X_train[i].reshape(-1, 1), X_train[j].reshape(-1, 1), dist=euclidean)
            train_distance_matrix[i, j] = distance
            train_distance_matrix[j, i] = distance
    
    print("Computing DTW distance matrix for testing data...")
    for i in range(n_test_samples):
        if i % 10 == 0:
            print(f"Testing sample {i}/{n_test_samples}")
        for j in range(n_train_samples):
            distance, _ = fastdtw(X_test[i].reshape(-1, 1), X_train[j].reshape(-1, 1), dist=euclidean)
            test_distance_matrix[i, j] = distance
    
    return train_distance_matrix, test_distance_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Compute DTW distance matrices for X_train and X_test
print("Computing DTW distance matrices...")
dtw_distance_matrix_train, dtw_distance_matrix_test = compute_dtw_distance_matrix(X_train, X_test)

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
    X_train_target = dtw_distance_matrix_train[y_train == target_user]

    # Initialize and train the models
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

    # Fit the models
    oc_svm.fit(X_train_target)
    iso_forest.fit(X_train_target)
    lof.fit(X_train_target)

    # Predict using the One-Class SVM
    y_pred_oc_svm = oc_svm.predict(dtw_distance_matrix_test)
    y_pred_oc_svm_binary = np.where(y_pred_oc_svm == 1, target_user, -1)

    # Predict using the Isolation Forest
    y_pred_iso_forest = iso_forest.predict(dtw_distance_matrix_test)
    y_pred_iso_forest_binary = np.where(y_pred_iso_forest == 1, target_user, -1)

    # Predict using the Local Outlier Factor
    y_pred_lof = lof.predict(dtw_distance_matrix_test)
    y_pred_lof_binary = np.where(y_pred_lof == 1, target_user, -1)

    # Evaluate One-Class SVM
    accuracy_oc_svm = accuracy_score(y_test == target_user, y_pred_oc_svm_binary == target_user)
    roc_auc_oc_svm = roc_auc_score(y_test == target_user, y_pred_oc_svm)
    cm_oc_svm = confusion_matrix(y_test == target_user, y_pred_oc_svm_binary == target_user)
    class_report_oc_svm = classification_report(y_test == target_user, y_pred_oc_svm_binary == target_user, output_dict=True)
    
    results.append({
        "model": "One-Class SVM",
        "target_user": target_user,
        "accuracy": accuracy_oc_svm,
        "roc_auc": roc_auc_oc_svm,
        "confusion_matrix": cm_oc_svm,
        "classification_report": class_report_oc_svm
    })

    print(f"User {target_user}: One-Class SVM Accuracy: {accuracy_oc_svm*100:.2f}%")
    print(f"User {target_user}: One-Class SVM ROC AUC: {roc_auc_oc_svm:.2f}")
    print(f"User {target_user}: One-Class SVM Classification Report:")
    print(pd.DataFrame(class_report_oc_svm).transpose())
    print(f"User {target_user}: One-Class SVM Confusion Matrix:")
    plot_confusion_matrix(cm_oc_svm, classes=['Non-Target', 'Target'])

    # Evaluate Isolation Forest
    accuracy_iso_forest = accuracy_score(y_test == target_user, y_pred_iso_forest_binary == target_user)
    roc_auc_iso_forest = roc_auc_score(y_test == target_user, y_pred_iso_forest)
    cm_iso_forest = confusion_matrix(y_test == target_user, y_pred_iso_forest_binary == target_user)
    class_report_iso_forest = classification_report(y_test == target_user, y_pred_iso_forest_binary == target_user, output_dict=True)
    
    results.append({
        "model": "Isolation Forest",
        "target_user": target_user,
        "accuracy": accuracy_iso_forest,
        "roc_auc": roc_auc_iso_forest,
        "confusion_matrix": cm_iso_forest,
        "classification_report": class_report_iso_forest
    })

    print(f"User {target_user}: Isolation Forest Accuracy: {accuracy_iso_forest*100:.2f}%")
    print(f"User {target_user}: Isolation Forest ROC AUC: {roc_auc_iso_forest:.2f}")
    print(f"User {target_user}: Isolation Forest Classification Report:")
    print(pd.DataFrame(class_report_iso_forest).transpose())
    print(f"User {target_user}: Isolation Forest Confusion Matrix:")
    plot_confusion_matrix(cm_iso_forest, classes=['Non-Target', 'Target'])

    # Evaluate Local Outlier Factor
    accuracy_lof = accuracy_score(y_test == target_user, y_pred_lof_binary == target_user)
    roc_auc_lof = roc_auc_score(y_test == target_user, y_pred_lof)
    cm_lof = confusion_matrix(y_test == target_user, y_pred_lof_binary == target_user)
    class_report_lof = classification_report(y_test == target_user, y_pred_lof_binary == target_user, output_dict=True)
    
    results.append({
        "model": "Local Outlier Factor",
        "target_user": target_user,
        "accuracy": accuracy_lof,
        "roc_auc": roc_auc_lof,
        "confusion_matrix": cm_lof,
        "classification_report": class_report_lof
    })

    print(f"User {target_user}: Local Outlier Factor Accuracy: {accuracy_lof*100:.2f}%")
    print(f"User {target_user}: Local Outlier Factor ROC AUC: {roc_auc_lof:.2f}")
    print(f"User {target_user}: Local Outlier Factor Classification Report:")
    print(pd.DataFrame(class_report_lof).transpose())
    print(f"User {target_user}: Local Outlier Factor Confusion Matrix:")
    plot_confusion_matrix(cm_lof, classes=['Non-Target', 'Target'])

# Print summary results
for result in results:
    print(f"Model: {result['model']}, User {result['target_user']} - Accuracy: {result['accuracy']*100:.2f}%")
    print(f"Model: {result['model']}, User {result['target_user']} - ROC AUC: {result['roc_auc']:.2f}")
    print(pd.DataFrame(result['classification_report']).transpose())
    print(result['confusion_matrix'])
