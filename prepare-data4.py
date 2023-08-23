import pandas as pd
import numpy as np


def alignColumnNames(df):
    df.rename(
        columns={
            "α-地中海（点突变型）": "alpha-1",
            "α-地中（点突变型）": "alpha-1",
            "α-地中海（缺失型）": "alpha-2",
            " β-地中海": "beta",
            "β-地中海": "beta",
            "RDW-CV（11.5-14.5）": "RDW-CV",
            "RDW-SD（39-46）": "RDW-SD",
            "MCV（82-100）": "MCV",
            "MCH（27-34）": "MCH",
            "MCHC（316-354）": "MCHC",
            "HGB(115-150": "HGB",
            "RDW-RD": "RDW-SD",
            "HBG": "HGB",
            "HCT（0.35-0.45）": "HCT",
            "HCT(0.35-0.45)": "HCT",
            "α-地中海贫血基因（点突变型）": "alpha-1",
            "α-地中海贫血基因（缺失型）": "alpha-2",
            "β-地中海贫血基因": "beta",
        },
        inplace=True,
    )
    df = df[
        [
            "年龄",
            "性别",
            "alpha-1",
            "alpha-2",
            "beta",
            "RBC",
            "RDW-CV",
            "RDW-SD",
            "MCV",
            "MCH",
            "MCHC",
            "HCT",
            "HGB",
        ]
    ]
    df = df.fillna({"alpha-1": "ab", "alpha-2": "ab", "beta": "ab", "年龄": "0岁"})
    df = df.dropna()
    # df = df.drop_duplicates().reset_index(drop=True)
    return df


def label_rows(row):
    if (
        row["alpha-1"] == "αα/αα"
        and row["alpha-2"] == "αα/αα"
        and (row["beta"] == "N / N" or row["beta"] == "N/N")
    ):
        return 0
    else:
        return 1


def cal_new_columns(df):
    df["RDW-CV/RBC"] = df["RDW-CV"] / df["RBC"]
    df["RDW-SD/RBC"] = df["RDW-SD"] / df["RBC"]
    df["MCV/RBC"] = df["MCV"] / df["RBC"]
    df["MCH/RBC"] = df["MCH"] / df["RBC"]
    df["MCHC/RBC"] = df["MCHC"] / df["RBC"]
    # df["HCT/RBC"] = df["HCT"] / df["RBC"]
    # df["HGB/RBC"] = df["HGB"] / df["RBC"]
    return df


def parseSex(row):
    if row["性别"] == "男":
        return 0.0
    elif row["性别"] == "女":
        return 1.0


def preprocess(df):
    df = alignColumnNames(df)
    df = cal_new_columns(df)
    df["label"] = df.apply(lambda row: label_rows(row), axis=1)
    df["Sex"] = df.apply(lambda row: parseSex(row), axis=1)
    df.drop(columns=["alpha-1", "alpha-2", "beta", "性别", "年龄"], inplace=True)
    return df


d = pd.read_csv("./data/origin/g-0819.csv")

d = preprocess(d)

print(d.shape)
print(f"all: {np.bincount(d['label'].values)}")

d.to_csv("./data/select/g-0819.csv")


d = pd.read_csv("./data/origin/g-0819-0.csv")

d = preprocess(d)
d = d.drop(columns=["Sex"])

print(d.shape)
print(f"male: {np.bincount(d['label'].values)}")

d.to_csv("./data/select/g-0819-0.csv")


d = pd.read_csv("./data/origin/g-0819-1.csv")

d = preprocess(d)
d = d.drop(columns=["Sex"])

print(d.shape)
print(f"female: {np.bincount(d['label'].values)}")

d.to_csv("./data/select/g-0819-1.csv")
