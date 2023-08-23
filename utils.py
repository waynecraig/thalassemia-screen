import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def get_data(path):
    df = pd.read_csv(path)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
            "Sex",
        ]
    ]

    if path != "./data/select/s.csv":
        X = X.drop(columns=["Sex"])

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X, y, X_train_std, X_test_std, y_train, y_test


def get_data_2(path):
    df = pd.read_csv(path)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
            "label-2",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    X_train = X_train.drop(columns=["label-2"])

    X_test1 = X_test[X_test["label-2"] == 2]
    y_test1 = y_test[X_test["label-2"] == 2]
    X_test2 = X_test[X_test["label-2"] != 2]
    y_test2 = y_test[X_test["label-2"] != 2]

    X_test1 = X_test1.drop(columns=["label-2"])
    X_test2 = X_test2.drop(columns=["label-2"])

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test1_std = sc.transform(X_test1)
    X_test2_std = sc.transform(X_test2)

    return X, X_train_std, X_test1_std, X_test2_std, y_train, y_test1, y_test2


def get_data_3(path):
    df = pd.read_csv(path)

    label_counts = df["label"].value_counts()
    count_diff = label_counts[0] - label_counts[1]
    if count_diff > 0:
        df = df.drop(df[df["label"] == 0].sample(count_diff).index)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X, X_train_std, X_test_std, y_train, y_test


def get_data_4(path):
    df = pd.read_csv(path)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X, X_train_std, X_test_std, y_train, y_test


def get_data_5(path):
    df = pd.read_csv(path)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    oversample = RandomOverSampler()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X, X_train_std, X_test_std, y_train, y_test


def get_data_6(path):
    df = pd.read_csv(path)

    X = df[
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
            "HCT/RBC",
            "HGB/RBC",
        ]
    ]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=37, stratify=y
    )

    undersample = RandomUnderSampler()
    X_train, y_train = undersample.fit_resample(X_train, y_train)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X, X_train_std, X_test_std, y_train, y_test


def get_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "kappa": cohen_kappa_score(y_test, y_pred),
    }
