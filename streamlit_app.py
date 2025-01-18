import streamlit as st
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    balanced_accuracy_score, 
    roc_curve, 
    auc
)

# --------------------------
#       PAGE TITLE
# --------------------------
st.title("Interactive Logistic Regression Demonstration")

st.write("""
This app demonstrates how Logistic Regression behaves with a mixture of *real* (informative) and *noisy* (uninformative) features.
Experiment with different parameters in the sidebar to see how the performance changes.
""")

# --------------------------
#    SIDEBAR CONTROLS
# --------------------------
st.sidebar.header("Data Generation Parameters")

n_real_features = st.sidebar.slider(
    "Number of Real (Informative) Features", 
    min_value=1, 
    max_value=10, 
    value=2, 
    help="How many features actually carry signal related to the target."
)

percent_real_features = st.sidebar.slider(
    "Percent of Real Features", 
    min_value=1, 
    max_value=100, 
    value=50, 
    step=5, 
    help="What percent of the total features are real vs. noisy."
)

n_samples = st.sidebar.slider(
    "Number of Samples", 
    min_value=50, 
    max_value=2000, 
    step=50, 
    value=200, 
    help="Number of data samples to generate."
)

st.sidebar.header("Model Parameters")

regularization_strength = st.sidebar.slider(
    "Regularization Strength (C)", 
    min_value=0.01, 
    max_value=10.0, 
    step=0.01, 
    value=1.0, 
    help="Inverse of regularization strength in Logistic Regression (smaller is more regularization)."
)

penalty_type = st.sidebar.selectbox(
    "Penalty Type", 
    ("l2", "none"), 
    help="Type of regularization to use in Logistic Regression. 'l2' is default; 'none' means no regularization."
)

scale_features = st.sidebar.checkbox(
    "Apply Feature Scaling", 
    value=True, 
    help="Standardizes features by removing the mean and scaling to unit variance."
)

test_size = st.sidebar.slider(
    "Test Size (%)", 
    min_value=10, 
    max_value=50, 
    step=5, 
    value=20, 
    help="Percentage of data to use for the test set."
)

# --------------------------
#     DATA GENERATION
# --------------------------
np.random.seed(42)

# Binary outcome
y = np.random.choice([0, 1], size=n_samples)

# Derived parameters
n_total_features = max(int(n_real_features / (percent_real_features / 100)), 1)
n_noisy_features = n_total_features - n_real_features if n_total_features > n_real_features else 0

# Generate real (informative) features
# A simple approach: the real features have some correlation with y 
# by shifting or scaling around the target.
real_features = np.column_stack([
    y + 0.8 * np.random.normal(size=n_samples) for _ in range(n_real_features)
])

# Generate noisy (uninformative) features
noisy_features = np.random.normal(size=(n_samples, n_noisy_features))

# Combine
X = np.hstack((real_features, noisy_features))

# Optional: Convert to DataFrame for easier manipulation & correlation analysis
feature_names_real = [f"Real_{i}" for i in range(1, n_real_features + 1)]
feature_names_noisy = [f"Noisy_{j}" for j in range(1, n_noisy_features + 1)]
all_feature_names = feature_names_real + feature_names_noisy

df = pd.DataFrame(X, columns=all_feature_names)
df["target"] = y

# --------------------------
#  TRAIN-TEST SPLIT & SCALING
# --------------------------
test_ratio = test_size / 100.0
X_train, X_test, y_train, y_test = train_test_split(
    df[all_feature_names], 
    df["target"], 
    test_size=test_ratio, 
    random_state=42
)

# Apply Feature Scaling if selected
if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    X_train = X_train.values
    X_test = X_test.values

# --------------------------
#    MODEL TRAINING
# --------------------------
model = LogisticRegression(
    C=regularization_strength, 
    penalty=penalty_type if penalty_type != 'none' else 'l2',  # 'none' overrides penalty
    solver='lbfgs' if penalty_type != 'none' else 'lbfgs', 
    max_iter=1000
)
# If 'none' is selected, set penalty to none
if penalty_type == 'none':
    model.set_params(penalty='none')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# --------------------------
#   PERFORMANCE METRICS
# --------------------------
st.subheader("Results")

col1, col2, col3 = st.columns(3)
col1.metric("Balanced Accuracy", f"{balanced_acc:.3f}")
col2.metric("Real Features", n_real_features)
col3.metric("Total Features", n_total_features)

st.write(f"**Number of Samples:** {n_samples}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# --------------------------
#   ROC CURVE & AUC
# --------------------------
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("Receiver Operating Characteristic")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# --------------------------
#   FEATURE IMPORTANCE
# --------------------------
# For logistic regression, feature importance can be seen via coefficients
if penalty_type != 'none':
    coefs = model.coef_[0]
else:
    # If penalty='none', no regularization is used; coefficients are still available
    coefs = model.coef_[0]

# Create a DataFrame for the feature importances
feature_importances = pd.DataFrame({
    "Feature": all_feature_names, 
    "Coefficient": coefs
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Feature Importance (Logistic Regression Coefficients)")
st.write("""
Features with higher positive coefficients push the model toward predicting the positive class (1),
while more negative coefficients push toward the negative class (0).
""")

fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
sns.barplot(data=feature_importances, x="Coefficient", y="Feature", ax=ax_imp, palette="viridis")
ax_imp.axvline(x=0, color='gray', linestyle='--')
ax_imp.set_title("Logistic Regression Coefficients")
st.pyplot(fig_imp)

# --------------------------
#   CORRELATION ANALYSIS
# --------------------------
with st.expander("Show Correlation Heatmap (Real vs. Noisy Features)"):
    st.write("Correlation matrix of all features and the target.")
    corr_matrix = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", ax=ax_corr, center=0)
    ax_corr.set_title("Correlation Heatmap")
    st.pyplot(fig_corr)

st.write("---")
st.write("Feel free to adjust the parameters in the sidebar and observe how these plots and metrics change!")
