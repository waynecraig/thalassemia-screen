import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from dfs import DFSClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import umap
import umap.plot
import os
from sklearn.utils import resample
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data_name = "g-0819"
data_desc = "MCV>80 All"
result_path = f"data/result/DFS/{data_name}"
os.mkdir(result_path)

data = pd.read_csv(f"data/select/{data_name}.csv")

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

ids_train = X_train["id"]
ids_test = X_test["id"]

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

dfs_classifier = DFSClassifier(type="all")

dfs_classifier.fit(X_train, y_train)

y_pred = dfs_classifier.predict(X_test)

ids_test_0_0 = ids_test[(y_test == 0) & (y_pred == 0)]
ids_test_0_1 = ids_test[(y_test == 0) & (y_pred == 1)]
ids_test_1_0 = ids_test[(y_test == 1) & (y_pred == 0)]
ids_test_1_1 = ids_test[(y_test == 1) & (y_pred == 1)]
ids_train_0 = ids_train[y_train == 0]
ids_train_1 = ids_train[y_train == 1]

ids_test_0_0.to_csv(f"{result_path}/ids_test_0_0.csv", index=False)
ids_test_0_1.to_csv(f"{result_path}/ids_test_0_1.csv", index=False)
ids_test_1_0.to_csv(f"{result_path}/ids_test_1_0.csv", index=False)
ids_test_1_1.to_csv(f"{result_path}/ids_test_1_1.csv", index=False)
ids_train_0.to_csv(f"{result_path}/ids_train_0.csv", index=False)
ids_train_1.to_csv(f"{result_path}/ids_train_1.csv", index=False)

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision: %.2f" % precision_score(y_test, y_pred))
print("Recall: %.2f" % recall_score(y_test, y_pred))
print("F1: %.2f" % f1_score(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print("Specificity: %.2f" % specificity)
print("Sensitivity: %.2f" % sensitivity)

# draw ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

tprs = []
aucs = []
n_bootstraps = 1000
rng = np.random.RandomState(0)
for i in range(n_bootstraps):
    y_test_bootstrap, y_pred_bootstrap = resample(y_test, y_pred, random_state=i)
    fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_test_bootstrap, y_pred_bootstrap)
    tprs.append(np.interp(fpr, fpr_bootstrap, tpr_bootstrap))
    tprs[-1][0] = 0.0
    roc_auc_bootstrap = auc(fpr_bootstrap, tpr_bootstrap)
    aucs.append(roc_auc_bootstrap)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
lower_bound = np.percentile(aucs, 2.5)
upper_bound = np.percentile(aucs, 97.5)

plt.figure()
plt.plot(fpr, tpr, color="b", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.fill_between(
    fpr,
    mean_tprs - 1.96 * std,
    mean_tprs + 1.96 * std,
    color="grey",
    alpha=0.3,
    label=f"95% CI ({lower_bound:.2f}-{upper_bound:.2f})",
)
plt.plot([0, 1], [0, 1], color="r", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {data_desc}")
plt.legend(loc="lower right")
plt.savefig(f"{result_path}/roc_with_ci", dpi=300)
plt.close()

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "specificity": specificity,
    "sensitivity": sensitivity,
    "roc_auc": roc_auc,
    "auc_ci_lower": lower_bound,
    "auc_ci_upper": upper_bound,
}

metrics = pd.DataFrame(metrics, index=[0])
metrics.to_csv(f"{result_path}/metrics.csv", index=False)

# draw confusion matrix
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{result_path}/cm", dpi=300)
plt.close()


def result_row(row):
    if row.name in ids_train_0.values:
        return "Train Nagative"
    if row.name in ids_train_1.values:
        return "Train Positive"
    if row.name in ids_test_0_0.values:
        return "Test True Nagative"
    if row.name in ids_test_0_1.values:
        return "Test False Positive"
    if row.name in ids_test_1_0.values:
        return "Test False Nagative"
    if row.name in ids_test_1_1.values:
        return "Test True Positive"
    return "ignore"


original = pd.read_csv(f"data/origin/{data_name}.csv")
original = original.loc[:, ~original.columns.str.contains("^Unnamed")]
original["result"] = original.apply(lambda row: result_row(row), axis=1)
original.to_csv(f"{result_path}/result.csv", index=False)

o_X = original[
    [
        "RBC",
        "RDW-CV",
        "RDW-SD",
        "MCV",
        "MCH",
        "MCHC",
        "HCT",
        "HGB",
        "RDW-CV/RBC",
        "RDW-SD/RBC",
        "MCV/RBC",
        "MCH/RBC",
        "MCHC/RBC",
        "result",
    ]
]
o_X = o_X.dropna()
o_y = o_X["result"]
o_X = o_X.drop(columns=["result"])

sc = StandardScaler()
sc.fit(o_X)
o_X = sc.transform(o_X)

color_key = {
    "Train Nagative": "green",
    "Train Positive": "yellow",
    "Test True Nagative": "blue",
    "Test False Positive": "purple",
    "Test False Nagative": "red",
    "Test True Positive": "orange",
    "ignore": "black",
}

mapper = umap.UMAP().fit(o_X)
umap.plot.points(
    mapper,
    labels=o_y,
    color_key=color_key,
    theme="fire",
)
plt.savefig(f"{result_path}/umap-n15", dpi=300)
plt.close()

# draw compare roc
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

oversampler = SMOTE(random_state=37)
X_train_std_s, y_train_s = oversampler.fit_resample(X_train_std, y_train)

rf_classifier = RandomForestClassifier(
    criterion="entropy", n_estimators=100, random_state=1, n_jobs=6
)

lr_classifier = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs", multi_class="auto", random_state=1
)

svm_classifier = SVC(kernel="rbf", C=1.0, random_state=1, probability=True)

voting_classifier = VotingClassifier(
    estimators=[("rf", rf_classifier), ("lr", lr_classifier), ("svm", svm_classifier)],
    voting="soft",
)

voting_classifier.fit(X_train_std_s, y_train_s)
# voting_classifier.fit(X_train_std, y_train)
voting_labels = voting_classifier.predict(X_test_std)
voting_probs = voting_classifier.predict_proba(X_test_std)[:, 1]

voting_fpr, voting_tpr, _ = roc_curve(y_test, voting_probs)
voting_auc = auc(voting_fpr, voting_tpr)

plt.plot(voting_fpr, voting_tpr, label="Voting (AUC = %0.2f)" % voting_auc)
plt.plot(fpr, tpr, label="DFS (AUC = %0.2f)" % roc_auc)

plt.plot([0, 1], [0, 1], "k--")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for {data_desc}")
plt.legend(loc="lower right")

plt.savefig(f"{result_path}/roc", dpi=300)
plt.close()


def calculate_IDI(predicted_probs_1, predicted_probs_0, true_labels):
    c1 = np.mean(predicted_probs_1[true_labels == 1])
    c0 = np.mean(predicted_probs_0[true_labels == 1])
    e1 = np.mean(predicted_probs_1[true_labels == 0])
    e0 = np.mean(predicted_probs_0[true_labels == 0])
    idi = (c1 - c0) - (e1 - e0)
    return idi


def calculate_NRI(predicted_labels_1, predicted_labels_0, true_labels):
    p1 = accuracy_score(true_labels, predicted_labels_1)
    p0 = accuracy_score(true_labels, predicted_labels_0)
    n1 = accuracy_score(1 - true_labels, 1 - predicted_labels_1)
    n0 = accuracy_score(1 - true_labels, 1 - predicted_labels_0)
    nri = (p1 - p0) + (n1 - n0)
    return nri


idi = calculate_IDI(voting_probs, y_pred, y_test)
nri = calculate_NRI(voting_labels, y_pred, y_test)

print("IDI: %.2f" % idi)
print("NRI: %.2f" % nri)

comparison = {
    "IDI": idi,
    "NRI": nri,
}
comparison = pd.DataFrame(comparison, index=[0])
comparison.to_csv(f"{result_path}/comparison.csv", index=False)
