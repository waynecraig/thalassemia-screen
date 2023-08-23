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
    df = df.fillna({"alpha-1": "ab", "alpha-2": "ab", "beta": "ab"})
    df = df.dropna().reset_index(drop=True)
    return df


def label_rows(row):
    if (
        row["alpha-1"] == "αα/αα"
        and row["alpha-2"] == "αα/αα"
        and row["beta"] == "N / N"
    ):
        return 0
    else:
        return 1


def label_rows_2(row):
    if row["MCV"] > 80:
        # if row["RDW-CV"] / row["RBC"] > 3.54:
        #     return 0
        # return 1
        return 3
    else:
        return 2


def cal_new_columns(df):
    df["RDW-CV/RBC"] = df["RDW-CV"] / df["RBC"]
    df["RDW-SD/RBC"] = df["RDW-SD"] / df["RBC"]
    df["MCV/RBC"] = df["MCV"] / df["RBC"]
    df["MCH/RBC"] = df["MCH"] / df["RBC"]
    df["MCHC/RBC"] = df["MCHC"] / df["RBC"]
    df["HCT/RBC"] = df["HCT"] / df["RBC"]
    df["HGB/RBC"] = df["HGB"] / df["RBC"]
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
    df["label-2"] = df.apply(lambda row: label_rows_2(row), axis=1)
    df["Sex"] = df.apply(lambda row: parseSex(row), axis=1)
    df.drop(columns=["alpha-1", "alpha-2", "beta", "性别"], inplace=True)
    return df


df_new_1 = pd.read_csv("./data/origin/new-1.csv")
df_new_2 = pd.read_csv("./data/origin/new-2.csv")
df_a = pd.read_csv("./data/origin/abnormal.csv")
df_n_1 = pd.read_csv("./data/origin/normal-1.csv")
df_n_2 = pd.read_csv("./data/origin/normal-2.csv")
df_n_3 = pd.read_csv("./data/origin/normal-3.csv")
df_n_bc = pd.read_csv("./data/origin/normal-body-check.csv")
df_n_p = pd.read_csv("./data/origin/normal-patient.csv")
df_m8_1 = pd.read_csv("./data/origin/m8_train/总MCV>80,HBG<120-Table 1.csv")
df_m8_2 = pd.read_csv("./data/origin/m8_train/总MCV>80,HBG>-120-Table 1.csv")
df_m8_3 = pd.read_csv("./data/origin/m8_test/总MCV>80,HBG<120-Table 1.csv")
df_m8_4 = pd.read_csv("./data/origin/m8_test/总MCV>80,HBG>=120-Table 1.csv")
df_j = pd.read_csv("./data/origin/j-0728.csv")

d1 = preprocess(df_new_1)
d2 = preprocess(df_new_2)
d3 = preprocess(df_a)
d4 = preprocess(df_n_1)
d5 = preprocess(df_n_2)
d6 = preprocess(df_n_3)
d7 = preprocess(df_n_bc)
d8 = preprocess(df_n_p)
d9 = preprocess(df_m8_1)
d10 = preprocess(df_m8_2)
d11 = preprocess(df_m8_3)
d12 = preprocess(df_m8_4)
d13 = preprocess(df_j)

d = pd.concat([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13], axis=0).reset_index(
    drop=True
)

print(d.shape)

d = d.drop_duplicates().reset_index(drop=True)

print(d.shape)

d.to_csv("./data/select/s.csv", index=False)
print(f"all: {np.bincount(d['label'].values)}")

gs = d.groupby(["Sex", "label-2"])

for name, group in gs:
    group.to_csv(f"./data/select/s-{name[0]:.0f}-{name[1]}.csv", index=False)
    print(f"{name[0]:.0f}-{name[1]}: {np.bincount(group['label'].values)}")

gs = d.groupby("Sex")

for name, group in gs:
    group.to_csv(f"./data/select/s-{name:.0f}.csv", index=False)
    print(f"{name:.0f}: {np.bincount(group['label'].values)}")

gs = d.groupby("label-2")

for name, group in gs:
    group.to_csv(f"./data/select/s-2-{name}.csv", index=False)
    print(f"{name}: {np.bincount(group['label'].values)}")
