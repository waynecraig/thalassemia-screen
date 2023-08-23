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
    df["Sex"] = df.apply(lambda row: parseSex(row), axis=1)
    df.drop(columns=["alpha-1", "alpha-2", "beta", "性别"], inplace=True)
    return df


df_train_1 = pd.read_csv("./data/origin/m8_train/总MCV>80,HBG<120-Table 1.csv")
df_train_2 = pd.read_csv("./data/origin/m8_train/总MCV>80,HBG>-120-Table 1.csv")
df_test_1 = pd.read_csv("./data/origin/m8_test/总MCV>80,HBG<120-Table 1.csv")
df_test_2 = pd.read_csv("./data/origin/m8_test/总MCV>80,HBG>=120-Table 1.csv")

df_train_1 = preprocess(df_train_1)
df_train_2 = preprocess(df_train_2)
df_test_1 = preprocess(df_test_1)
df_test_2 = preprocess(df_test_2)

d = pd.concat([df_train_1, df_train_2, df_test_1, df_test_2], axis=0).reset_index(
    drop=True
)
d = d.drop_duplicates().reset_index(drop=True)

d1 = pd.concat([df_train_1, df_test_1], axis=0).reset_index(drop=True)
d1 = d1.drop_duplicates().reset_index(drop=True)

d2 = pd.concat([df_train_2, df_test_2], axis=0).reset_index(drop=True)
d2 = d2.drop_duplicates().reset_index(drop=True)

# df_ptest_1 = pd.merge(df_test_1, df_train_1, how="outer", indicator=True)
# df_ptest_1 = df_ptest_1[df_ptest_1["_merge"] == "left_only"].drop('_merge', axis=1)
# df_ptest_2 = pd.merge(df_test_2, df_train_2, how="outer", indicator=True)
# df_ptest_2 = df_ptest_2[df_ptest_2["_merge"] == "left_only"].drop('_merge', axis=1)

print(d.shape)
print(f"all: {np.bincount(d['label'].values)}")

print(d1.shape)
print(f"HGB<120: {np.bincount(d1['label'].values)}")
print(f"HGB<120,train: {np.bincount(df_train_1['label'].values)}")
print(f"HGB<120,test: {np.bincount(df_test_1['label'].values)}")

print(d2.shape)
print(f"HGB>=120: {np.bincount(d2['label'].values)}")
print(f"HGB>=120,train: {np.bincount(df_train_2['label'].values)}")
print(f"HGB>=120,test: {np.bincount(df_test_2['label'].values)}")

m1 = d[d["HGB"] < 120]
print(m1.shape)
print(f"HGB<120,auto: {np.bincount(m1['label'].values)}")
m2 = d[d["HGB"] >= 120]
print(m2.shape)
print(f"HGB>=120,auto: {np.bincount(m2['label'].values)}")

# print(df_train_1.shape)
# print(df_train_2.shape)
# print(df_test_1.shape)
# print(df_test_2.shape)
# print(df_ptest_1.shape)
# print(df_ptest_2.shape)

# merged_1 = pd.merge(df_train_1, df_test_1, how="inner", on=list(df_train_1.columns))
# merged_2 = pd.merge(df_train_2, df_test_2, how="inner", on=list(df_train_2.columns))
# print(merged_1)
# print(merged_2)

d.to_csv("./data/select/m8.csv", index=False)
d1.to_csv("./data/select/m8_1.csv", index=False)
d2.to_csv("./data/select/m8_2.csv", index=False)

# df_train_1.to_csv("./data/select/m8_train_1.csv", index=False)
# df_train_2.to_csv("./data/select/m8_train_2.csv", index=False)
# df_test_1.to_csv("./data/select/m8_test_1.csv", index=False)
# df_test_2.to_csv("./data/select/m8_test_2.csv", index=False)
# df_ptest_1.to_csv("./data/select/m8_ptest_1.csv", index=False)
# df_ptest_2.to_csv("./data/select/m8_ptest_2.csv", index=False)
