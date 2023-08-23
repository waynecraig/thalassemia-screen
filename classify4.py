import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE

data = pd.read_csv("data/select/j-0801-0.csv")

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

ids_train = X_train["id"]
ids_test = X_test["id"]

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X_train_std)
# df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "target": y_train})
# sns.scatterplot(data=df_tsne, x="x", y="y", hue="target")
# plt.title("t-SNE before SMOTE")
# plt.savefig("data/j/tsne-train.png", dpi=300)
# plt.close()

oversampler = SMOTE(random_state=37)
# oversampler = RandomOverSampler(random_state=37)
X_train_std, y_train = oversampler.fit_resample(X_train_std, y_train)

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X_train_std)
# df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "target": y_train})
# sns.scatterplot(data=df_tsne, x="x", y="y", hue="target")
# plt.title("t-SNE after SMOTE")
# plt.savefig("data/j/tsne-train-over.png", dpi=300)
# plt.close()

classifier = RandomForestClassifier(
    criterion="entropy", n_estimators=100, random_state=1, n_jobs=6
)

# classifier = GaussianNB()

classifier.fit(X_train_std, y_train)

y_pred = classifier.predict(X_test_std)

ids_test_0_0 = ids_test[(y_test == 0) & (y_pred == 0)]
ids_test_0_1 = ids_test[(y_test == 0) & (y_pred == 1)]
ids_test_1_0 = ids_test[(y_test == 1) & (y_pred == 0)]
ids_test_1_1 = ids_test[(y_test == 1) & (y_pred == 1)]

ids_train.to_csv("data/j/ids_train.csv", index=False)
ids_test_0_0.to_csv("data/j/ids_test_0_0.csv", index=False)
ids_test_0_1.to_csv("data/j/ids_test_0_1.csv", index=False)
ids_test_1_0.to_csv("data/j/ids_test_1_0.csv", index=False)
ids_test_1_1.to_csv("data/j/ids_test_1_1.csv", index=False)

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

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "specificity": specificity,
    "sensitivity": sensitivity,
}

metrics = pd.DataFrame(metrics, index=[0])
metrics.to_csv("data/j/metrics.csv", index=False)

# draw confusion matrix
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("data/j/cm", dpi=300)
plt.close()

explainer = shap.TreeExplainer(classifier)
# explainer = shap.KernelExplainer(classifier.predict_proba, X_train_std)
shap_values = explainer.shap_values(X_test_std)
shap.initjs()
shap.summary_plot(shap_values[1], X_test_std, feature_names=X.columns[1:], show=False)
plt.savefig("data/j/shap/summary", dpi=300)
plt.close()

for i in range(0, len(X_test_std)):
    id = ids_test.iloc[i]
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][i],
        X_test_std[i],
        feature_names=X.columns[1:],
        matplotlib=True,
        show=False,
    )
    plt.savefig(f"data/j/shap/force-{id+2}", dpi=300)
    plt.close()


def result_row(row):
    if row.name in ids_train.values:
        return "train"
    if row.name in ids_test_0_0.values:
        return "test_0_0"
    if row.name in ids_test_0_1.values:
        return "test_0_1"
    if row.name in ids_test_1_0.values:
        return "test_1_0"
    if row.name in ids_test_1_1.values:
        return "test_1_1"
    return "ignore"


original = pd.read_csv("data/origin/j-0801-0.csv")
original["result"] = original.apply(lambda row: result_row(row), axis=1)
# original = original.drop(columns=["Unnamed: 20"])
original.to_csv("data/j/result.csv", index=False)
