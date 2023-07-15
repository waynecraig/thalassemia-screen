import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import os
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)
import seaborn as sns
from utils import get_data, get_metrics, get_data_3

paths = [
    "./data/select/s.csv",
    "./data/select/s-0.csv",
    "./data/select/s-1.csv",
    "./data/select/s-0-2.csv",
    "./data/select/s-1-2.csv",
    "./data/select/s-0-0.csv",
    "./data/select/s-1-0.csv",
    "./data/select/s-0-1.csv",
    "./data/select/s-1-1.csv",
    "./data/select/s-0-3.csv",
    "./data/select/s-1-3.csv",
    "./data/select/s-2-0.csv",
    "./data/select/s-2-1.csv",
]

models = [
    # (
    #     "lr",
    #     "Logistic Regression",
    # ),
    # (
    #     "svm",
    #     "Support Vector Machine",
    # ),
    # (
    #     "rf",
    #     "Random Forest",
    # ),
    # (
    #     "gbm",
    #     "Gradient Boosting Machine",
    # ),
    # (
    #     "voting",
    #     "Voting Classifier",
    # ),
    (
        "mlp",
        "Multi-layer Perceptron",
    ),
]


def get_classifiers(label):
    if label == "lr":
        return LogisticRegression(C=100.0, random_state=71, max_iter=500)
    elif label == "svm":
        return SVC(kernel="rbf", random_state=71, probability=False)
    elif label == "gbm":
        return GradientBoostingClassifier(criterion="friedman_mse", random_state=71)
    elif label == "rf":
        return RandomForestClassifier(
            criterion="entropy", n_estimators=10, random_state=71
        )
    elif label == "voting":
        return VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=100.0, random_state=71, max_iter=500)),
                ("svm", SVC(kernel="rbf", random_state=71, probability=True)),
                ("gbm", GradientBoostingClassifier()),
            ],
            voting="soft",
        )
    elif label == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100,), activation="relu", solver="lbfgs", max_iter=5000
        )


def get_data_name(filename):
    if filename == "s":
        return "全量数据"
    if filename == "s-0":
        return "男"
    if filename == "s-1":
        return "女"
    elif filename == "s-0-0":
        return "男 MCV>80 RDW-CV/RBC>3.54"
    elif filename == "s-0-1":
        return "男 MCV>80 RDW-CV/RBC<=3.54"
    elif filename == "s-0-2":
        return "男 MCV<=80"
    elif filename == "s-1-0":
        return "女 MCV>80 RDW-CV/RBC>3.54"
    elif filename == "s-1-1":
        return "女 MCV>80 RDW-CV/RBC<=3.54"
    elif filename == "s-1-2":
        return "女 MCV<=80"
    elif filename == "s-0-3":
        return "男 MCV>80"
    elif filename == "s-1-3":
        return "女 MCV>80"
    elif filename == "s-2-0":
        return "MCV>80 RDW-CV/RBC>3.54"
    elif filename == "s-2-1":
        return "MCV>80 RDW-CV/RBC<=3.54"


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
    for model in models:
        X, X_train_std, X_test_std, y_train, y_test = get_data(path)
        filename = os.path.splitext(os.path.basename(path))[0]

        classifier = get_classifiers(model[0])
        classifier.fit(X_train_std, y_train)
        y_pred = classifier.predict(X_test_std)
        metrics = get_metrics(y_test, y_pred)
        row = pd.DataFrame(
            {
                "file": [get_data_name(filename)],
                "model": [model[1]],
                "train_count": [np.bincount(y_train)],
                "test_count": [np.bincount(y_test)],
                "accuracy": [f"{metrics['accuracy']:.2f}"],
                "precision": [f"{metrics['precision']:.2f}"],
                "recall": [f"{metrics['recall']:.2f}"],
                "f1": [f"{metrics['f1']:.2f}"],
                "kappa": [f"{metrics['kappa']:.2f}"],
            }
        )

        # draw ROC curve
        if (
            model[0] == "dt"
            or model[0] == "rf"
            or model[0] == "knn"
            or model[0] == "mlp"
            or model[0] == "voting"
        ):
            probabilities = classifier.predict_proba(X_test_std)
            y_score = probabilities[:, 1]
        else:
            y_score = classifier.decision_function(X_test_std)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc_score = roc_auc_score(y_test, y_score)
        row["auc"] = f"{auc_score:.2f}"
        # plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc_score)
        # plt.plot([0, 1], [0, 1], "k--")  # Random guessing line
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("Receiver Operating Characteristic (ROC)")
        # plt.legend(loc="lower right")
        # plt.savefig(f"./data/result/s/{model[0]}-roc-{filename}.png", dpi=300)
        # plt.close()

        # draw confusion matrix
        # cm = confusion_matrix(y_test, y_pred)
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix")
        # plt.savefig(f"./data/result/s/{model[0]}-cm-{filename}.png", dpi=300)
        # plt.close()

        result = pd.concat([result, row], ignore_index=True)


result.to_csv("./data/result/s.csv", index=False)
