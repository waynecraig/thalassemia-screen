import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import umap
import umap.plot
import matplotlib.pyplot as plt
import numpy as np
import shap

df = pd.read_csv("./data/select/l-0819-1.csv")
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


rf_classifier.fit(X, y)
lr_classifier.fit(X, y)
svm_classifier.fit(X, y)
voting_classifier.fit(X, y)

testCases = pd.DataFrame()
testCases["RBC"] = [4.03, 4.34]
testCases["RDW-CV"] = [20.7, 23.5]
testCases["RDW-SD"] = [54.0, 61.1]
testCases["MCV"] = [71.7, 77.0]
testCases["MCH"] = [21.5, 20.7]
testCases["MCHC"] = [301, 269]
testCases["HCT"] = [0.289, 0.334]
testCases["HGB"] = [87, 90]
testCases["RDW-CV/RBC"] = [5.14, 5.41]
testCases["RDW-SD/RBC"] = [13.40, 14.08]
testCases["MCV/RBC"] = [17.79, 17.74]
testCases["MCH/RBC"] = [5.33, 4.77]
testCases["MCHC/RBC"] = [74.69, 61.98]

testCase_std = sc.transform(testCases)
rf_y = rf_classifier.predict(testCase_std)
lr_y = lr_classifier.predict(testCase_std)
svm_y = svm_classifier.predict(testCase_std)
voting_y = voting_classifier.predict(testCase_std)

print(f"rf: {rf_y}")
print(f"lr: {lr_y}")
print(f"svm: {svm_y}")
print(f"voting: {voting_y}")

explainer = shap.KernelExplainer(voting_classifier.predict_proba, X)
shap_values = explainer.shap_values(testCase_std)
shap.initjs()
for i in range(len(testCase_std)):
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][i],
        X[i],
        matplotlib=True,
        show=False,
        feature_names=testCases.columns,
    )
    plt.savefig(f"data/predict/shap-{i}", dpi=300)
    plt.close()

X = np.append(X, testCase_std, axis=0)
y = pd.concat([y, pd.Series([2, 3])], ignore_index=True)

# convert label code to label name
y = y.map(
    {0: "Train Nagative", 1: "Train Positive", 2: "Test Case 1", 3: "Test Case 2"}
)

color_key = {
    "Train Nagative": "#1f77b4",
    "Train Positive": "#2ca02c",
    "Test Case 1": "#ff7f0e",
    "Test Case 2": "#d62728",
}

mapper = umap.UMAP().fit(X)
umap.plot.points(
    mapper,
    labels=y,
    color_key=color_key,
    theme="fire",
)
plt.savefig("data/predict/umap", dpi=300)
plt.close()

result = testCases.copy()
result["rf_result"] = rf_y
result["lr_result"] = lr_y
result["svm_result"] = svm_y
result["voting_result"] = voting_y
result.to_csv("data/predict/result.csv", index=False)
