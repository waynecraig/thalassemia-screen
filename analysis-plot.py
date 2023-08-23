import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot

# 读取数据
# data = pd.read_csv("data/select/j-0801.csv")
# data = pd.read_csv("data/select/j-0801-0.csv")
# data = pd.read_csv("data/select/j-0801-1.csv")
data = pd.read_csv("data/j5-all-smote/result.csv")

X = data[
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
y = data["result"]

# 标准化特征向量
X = StandardScaler().fit_transform(X)

mapper = umap.UMAP(densmap=False).fit(X)
umap.plot.points(mapper, labels=y)
# umap.plot.points(mapper)
plt.show()
