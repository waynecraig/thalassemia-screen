import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv("data/select/j-0801-0.csv")

data.drop(
    # columns=["id", "RDW-CV/RBC", "RDW-SD/RBC", "MCV/RBC", "MCH/RBC", "MCHC/RBC"],
    columns=["id"],
    inplace=True,
)

data.dropna(inplace=True)

# 提取特征和目标变量
X = data.drop(columns=["label"], axis=1)
y = data["label"]

# 标准化特征向量
X = StandardScaler().fit_transform(X)

# 初始化t-SNE模型
tsne = TSNE(n_components=2, random_state=42)

# 转换特征向量
X_tsne = tsne.fit_transform(X)

# 创建一个包含特征向量和目标变量的DataFrame
df_tsne = pd.DataFrame({"x": X_tsne[:, 0], "y": X_tsne[:, 1], "target": y})

# 使用Seaborn绘制t-SNE图
sns.scatterplot(data=df_tsne, x="x", y="y", hue="target")
plt.title("t-SNE for j-0801 Male")
plt.savefig("data/tsne/j-0801-0", dpi=300)
