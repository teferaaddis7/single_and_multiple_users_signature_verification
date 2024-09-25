import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw

# Load the dataset
data_path = '/mnt/data/HnS_AllFeaturesWithHeader.csv'
df = pd.read_csv(data_path, delimiter=',')
print("Dataset loaded successfully:")
print(df.head())

# Label the data
n_users = 30
samples_per_user = 60
labels = np.repeat(np.arange(n_users), samples_per_user)
df['label'] = labels

# Split features and labels
X = df.iloc[:, :-1].values
y = df['label'].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to compute DTW distances between samples
def compute_dtw_matrix(X):
    n_samples = X.shape[0]
    dtw_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            distance, _ = fastdtw(X[i], X[j])
            dtw_matrix[i, j] = distance
            dtw_matrix[j, i] = distance
    return dtw_matrix

print("Computing DTW matrix...")
# Compute DTW distances matrix
dtw_matrix = compute_dtw_matrix(X_scaled)
print("DTW matrix computed successfully.")

# Use DTW distances as features
X_dtw = dtw_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dtw, y, test_size=0.3, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Initialize classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "SVC": SVC(kernel='linear', C=1),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Voting": VotingClassifier(estimators=[
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('svc', SVC(kernel='linear', C=1, probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ], voting='soft'),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}

# Train and evaluate classifiers
results = {}

print("Training and evaluating classifiers...")
for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    results[clf_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }
    print(f"{clf_name} trained successfully.")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Print results
print("Printing results...")
for clf_name, result in results.items():
    print(f"{clf_name} Accuracy: {result['accuracy']*100:.2f}%")
    print(f"{clf_name} Precision: {result['precision']*100:.2f}%")
    print(f"{clf_name} Recall: {result['recall']*100:.2f}%")
    print(f"{clf_name} F1-Score: {result['f1']*100:.2f}%")
    print(f"{clf_name} Confusion Matrix:\n{result['confusion_matrix']}\n")
    plot_confusion_matrix(result['confusion_matrix'], f'{clf_name} Confusion Matrix')

# Perform one-vs-all, one-vs-one, and output-code classification
ova_classifiers = {name: OneVsRestClassifier(clf) for name, clf in classifiers.items()}
ovo_classifiers = {name: OneVsOneClassifier(clf) for name, clf in classifiers.items()}
output_code_classifiers = {name: OutputCodeClassifier(clf, code_size=2, random_state=42) for name, clf in classifiers.items()}

# Function to train and evaluate multiclass classifiers
def evaluate_multiclass_classifiers(classifiers, X_train, y_train, X_test, y_test):
    results = {}
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        results[clf_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }
    return results

print("Evaluating One-vs-All classifiers...")
ova_results = evaluate_multiclass_classifiers(ova_classifiers, X_train, y_train, X_test, y_test)
for clf_name, result in ova_results.items():
    print(f"OVA {clf_name} Accuracy: {result['accuracy']*100:.2f}%")
    print(f"OVA {clf_name} Precision: {result['precision']*100:.2f}%")
    print(f"OVA {clf_name} Recall: {result['recall']*100:.2f}%")
    print(f"OVA {clf_name} F1-Score: {result['f1']*100:.2f}%")
    print(f"OVA {clf_name} Confusion Matrix:\n{result['confusion_matrix']}\n")
    plot_confusion_matrix(result['confusion_matrix'], f'OVA {clf_name} Confusion Matrix')

print("Evaluating One-vs-One classifiers...")
ovo_results = evaluate_multiclass_classifiers(ovo_classifiers, X_train, y_train, X_test, y_test)
for clf_name, result in ovo_results.items():
    print(f"OVO {clf_name} Accuracy: {result['accuracy']*100:.2f}%")
    print(f"OVO {clf_name} Precision: {result['precision']*100:.2f}%")
    print(f"OVO {clf_name} Recall: {result['recall']*100:.2f}%")
    print(f"OVO {clf_name} F1-Score: {result['f1']*100:.2f}%")
    print(f"OVO {clf_name} Confusion Matrix:\n{result['confusion_matrix']}\n")
    plot_confusion_matrix(result['confusion_matrix'], f'OVO {clf_name} Confusion Matrix')

print("Evaluating Output-Code classifiers...")
output_code_results = evaluate_multiclass_classifiers(output_code_classifiers, X_train, y_train, X_test, y_test)
for clf_name, result in output_code_results.items():
    print(f"Output-Code {clf_name} Accuracy: {result['accuracy']*100:.2f}%")
    print(f"Output-Code {clf_name} Precision: {result['precision']*100:.2f}%")
    print(f"Output-Code {clf_name} Recall: {result['recall']*100:.2f}%")
    print(f"Output-Code {clf_name} F1-Score: {result['f1']*100:.2f}%")
    print(f"Output-Code {clf_name} Confusion Matrix:\n{result['confusion_matrix']}\n")
    plot_confusion_matrix(result['confusion_matrix'], f'Output-Code {clf_name} Confusion Matrix')
