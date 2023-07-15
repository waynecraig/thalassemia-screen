from utils import get_data_2, get_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

paths = [
    "./data/select/s-0.csv",
    "./data/select/s-1.csv",
]


def get_data_name(filename, type):
    if filename == "s-0":
        if type == 1:
            return "男 MCV<=80"
        else:
            return "男 MCV>80"
    else:
        if type == 1:
            return "女 MCV<=80"
        else:
            return "女 MCV>80"


result = pd.DataFrame(
    columns=[
        "file",
        "model",
        "train_count",
        "test_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "kappa",
    ]
)


for path in paths:
    X, X_train_std, X_test1_std, X_test2_std, y_train, y_test1, y_test2 = get_data_2(
        path
    )
    filename = os.path.splitext(os.path.basename(path))[0]

    model = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=100.0, random_state=71, max_iter=500)),
            ("svm", SVC(kernel="rbf", random_state=71, probability=True)),
            ("gbm", GradientBoostingClassifier()),
        ],
        voting="soft",
    )

    model.fit(X_train_std, y_train)
    y_pred1 = model.predict(X_test1_std)
    y_pred2 = model.predict(X_test2_std)

    metrics = get_metrics(y_test1, y_pred1)
    row = pd.DataFrame(
        {
            "file": [get_data_name(filename, 1)],
            "model": ["voting"],
            "train_count": [np.bincount(y_train)],
            "test_count": [np.bincount(y_test1)],
            "accuracy": [f"{metrics['accuracy']:.2f}"],
            "precision": [f"{metrics['precision']:.2f}"],
            "recall": [f"{metrics['recall']:.2f}"],
            "f1": [f"{metrics['f1']:.2f}"],
            "kappa": [f"{metrics['kappa']:.2f}"],
        }
    )

    probabilities = model.predict_proba(X_test1_std)
    y_score = probabilities[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test1, y_score)
    auc_score = roc_auc_score(y_test1, y_score)
    row["auc"] = f"{auc_score:.2f}"

    result = pd.concat([result, row], ignore_index=True)

    metrics = get_metrics(y_test2, y_pred2)
    row = pd.DataFrame(
        {
            "file": [get_data_name(filename, 2)],
            "model": ["voting"],
            "train_count": [np.bincount(y_train)],
            "test_count": [np.bincount(y_test2)],
            "accuracy": [f"{metrics['accuracy']:.2f}"],
            "precision": [f"{metrics['precision']:.2f}"],
            "recall": [f"{metrics['recall']:.2f}"],
            "f1": [f"{metrics['f1']:.2f}"],
            "kappa": [f"{metrics['kappa']:.2f}"],
        }
    )

    probabilities = model.predict_proba(X_test2_std)
    y_score = probabilities[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test2, y_score)
    auc_score = roc_auc_score(y_test2, y_score)
    row["auc"] = f"{auc_score:.2f}"

    result = pd.concat([result, row], ignore_index=True)

    # cm1 = confusion_matrix(y_test1, y_pred1)
    # sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Female, MCV<=80")
    # plt.savefig(f"./data/result/s/cm-1-1.png", dpi=300)
    # plt.close()

    # cm2 = confusion_matrix(y_test2, y_pred2)
    # sns.heatmap(cm2, annot=True, fmt="d", cmap="Reds")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Female, MCV>80")
    # plt.savefig(f"./data/result/s/cm-1-2.png", dpi=300)
    # plt.close()


result.to_csv("./data/result/a.csv", index=False)
