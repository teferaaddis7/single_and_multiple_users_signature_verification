import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

# Split features and labels
X = df.iloc[:, :-1].values
y = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize base classifiers
base_classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=20),
    "SVC": SVC(kernel='linear', C=1),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "AdaBoost": AdaBoostClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Initialize one-vs-rest and one-vs-one classifiers
ovr_classifiers = {name: OneVsRestClassifier(clf) for name, clf in base_classifiers.items()}
ovo_classifiers = {name: OneVsOneClassifier(clf) for name, clf in base_classifiers.items()}

# Train and evaluate classifiers
results = {}

print("Training and evaluating classifiers...")
for strategy, classifiers in [("One-vs-Rest", ovr_classifiers), ("One-vs-One", ovo_classifiers)]:
    print(f"\n--- {strategy} Strategy ---")
    for clf_name, clf in classifiers.items():
        print(f"Training {clf_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results[(strategy, clf_name)] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": class_report
        }
        print(f"{clf_name} trained successfully.")
print("All classifiers trained and evaluated.")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to calculate TP, TN, FP, FN
def calculate_tp_tn_fp_fn(cm):
    TP = np.diag(cm).sum()
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return TP, TN.sum(), FP.sum(), FN.sum()

# Print results
for (strategy, clf_name), result in results.items():
    print(f"{strategy} - {clf_name} Accuracy: {result['accuracy']*100:.2f}%")
    print(f"{strategy} - {clf_name} Classification Report:")
    print(pd.DataFrame(result['classification_report']).transpose())
    print(f"{strategy} - {clf_name} Confusion Matrix:")
    plot_confusion_matrix(result['confusion_matrix'], classes=np.arange(n_users))
    
    # Calculate and print TP, TN, FP, FN
    TP, TN, FP, FN = calculate_tp_tn_fp_fn(result['confusion_matrix'])
    print(f"{strategy} - {clf_name} TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
