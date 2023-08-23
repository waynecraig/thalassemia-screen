import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("./data/select/s-1-2.csv")
# df = pd.read_csv("./data/select/s-1.csv")
# df = pd.read_csv("./data/select/s.csv")
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
    ]
]
y = df["label"]
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
# classifier = RandomForestClassifier(
#     criterion="entropy", n_estimators=100, random_state=37
# )
# classifier = SVC(kernel="rbf", random_state=37)
classifier = LogisticRegression(random_state=37)
classifier.fit(X, y)

testCase = pd.DataFrame()
# testCase["RBC"] = [4.03]
# testCase["RDW-CV"] = [20.7]
# testCase["RDW-SD"] = [54.0]
# testCase["MCV"] = [71.7]
# testCase["MCH"] = [21.5]
# testCase["MCHC"] = [301]
# testCase["HCT"] = [0.289]
# testCase["HGB"] = [87]
# testCase["RDW-CV/RBC"] = [5.14]
# testCase["RDW-SD/RBC"] = [13.40]
# testCase["MCV/RBC"] = [17.79]
# testCase["MCH/RBC"] = [5.33]
# testCase["MCHC/RBC"] = [74.69]

# testCase["RBC"] = [4.34]
# testCase["RDW-CV"] = [23.5]
# testCase["RDW-SD"] = [61.1]
# testCase["MCV"] = [77.0]
# testCase["MCH"] = [20.7]
# testCase["MCHC"] = [269]
# testCase["HCT"] = [0.334]
# testCase["HGB"] = [90]
# testCase["RDW-CV/RBC"] = [5.41]
# testCase["RDW-SD/RBC"] = [14.08]
# testCase["MCV/RBC"] = [17.74]
# testCase["MCH/RBC"] = [4.77]
# testCase["MCHC/RBC"] = [61.98]

testCase["RBC"] = [5.01]
testCase["RDW-CV"] = [12.6]
testCase["RDW-SD"] = [36.2]
testCase["MCV"] = [80.8]
testCase["MCH"] = [25.9]
testCase["MCHC"] = [321]
testCase["HCT"] = [0.405]
testCase["HGB"] = [130]
testCase["RDW-CV/RBC"] = [2.51]
testCase["RDW-SD/RBC"] = [7.22]
testCase["MCV/RBC"] = [16.13]
testCase["MCH/RBC"] = [5.17]
testCase["MCHC/RBC"] = [64.07]

testCase = sc.transform(testCase)
print(classifier.predict(testCase))
