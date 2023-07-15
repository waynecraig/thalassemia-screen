import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils import get_data


X, X_train_std, X_test_std, y_train, y_test = get_data("./data/select/s.csv")
model = LogisticRegression(C=100.0, random_state=71, max_iter=500).fit(
    X_train_std, y_train
)

explainer = shap.Explainer(model, X_train_std, feature_names=X.columns)
shap_values = explainer(X_test_std)

# shap.plots.beeswarm(shap_values)
# shap.plots.bar(shap_values)
shap.summary_plot(shap_values, X_test_std)
