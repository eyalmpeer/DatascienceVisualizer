import streamlit as st
import numpy as np
import pandas as pd

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

# Plotly for visualization
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# --------------------------
#       PAGE TITLE
# --------------------------
st.title("Interactive Logistic Regression Demonstration (Plotly Version)")

st.write("""
This app demonstrates how Logistic Regression behaves with a mixture of *real* (informative) and *noisy* (uninformative) features.
Use the sidebar to control:
- How many real (informative) vs. noisy features there are
- How large the dataset is
- How strongly the real features correlate with the outcome
- Model parameters like regularization strength, penalty type, and whether to scale features
- And more...
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

# New slider for correlation strength of real features
alpha = st.sidebar.slider(
    "Correlation Strength of Real Features",
    min_value=0.00,
    max_value=1.00,
    step=0.05,
    value=0.80,
    help="Controls how strongly real features correlate with the outcome (0=none, 1=strong)."
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
# Correlation strength is controlled by 'real_feature_correlation'
# The higher this is, the more separation between classes.
alpha = real_feature_correlation  # in [0,1] range
noise = np.random.normal(0, 1, size=n_samples)
real_features = np.column_stack([
    alpha * y + (1 - alpha) * noise
    for _ in range(n_real_features)
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
    penalty=penalty_type if penalty_type != 'none' else 'l2', 
    solver='lbfgs', 
    max_iter=1000
)
if penalty_type == 'none':
    # Overwrite penalty to 'none'
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
st.write(f"**Correlation Strength (Real Features):** {real_feature_correlation}")

# --------------------------
#   CONFUSION MATRIX
# --------------------------
cm = confusion_matrix(y_test, y_pred)

# Create a Plotly annotated heatmap
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted 0", "Predicted 1"],
    y=["Actual 0", "Actual 1"],
    colorscale="Blues",
    showscale=True,
    reversescale=False
)
cm_fig.update_layout(
    title="Confusion Matrix",
    margin=dict(l=50, r=50, t=50, b=50)
)
cm_fig.update_yaxes(autorange="reversed")  # So labels read top-to-bottom
st.plotly_chart(cm_fig, use_container_width=True)

# --------------------------
#   ROC CURVE & AUC
# --------------------------
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(
    x=fpr, 
    y=tpr, 
    mode='lines', 
    line=dict(color='darkorange', width=2),
    name=f"ROC curve (AUC = {roc_auc:.3f})"
))
roc_fig.add_shape(
    type="line", 
    x0=0, 
    y0=0, 
    x1=1, 
    y1=1,
    line=dict(color="navy", width=1, dash="dash")
)
roc_fig.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.6, y=0.05)
)
st.plotly_chart(roc_fig, use_container_width=True)

# --------------------------
#   FEATURE IMPORTANCE
# --------------------------
coefs = model.coef_[0]

feature_importances = pd.DataFrame({
    "Feature": all_feature_names, 
    "Coefficient": coefs
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Feature Importance (Logistic Regression Coefficients)")
st.write("""
Features with higher positive coefficients push the model toward predicting the positive class (1),
while more negative coefficients push toward the negative class (0).
""")

# Create a horizontal bar chart with Plotly
imp_fig = go.Figure()
imp_fig.add_trace(go.Bar(
    x=feature_importances["Coefficient"],
    y=feature_importances["Feature"],
    orientation='h',
    marker=dict(
        color=feature_importances["Coefficient"],
        colorscale='Viridis'
    ),
))
imp_fig.update_layout(
    title="Logistic Regression Coefficients",
    xaxis_title="Coefficient",
    yaxis_title="Feature",
    margin=dict(l=100, r=30, t=50, b=50)
)
# Draw a vertical line at x=0 for reference
imp_fig.add_shape(
    dict(
        type="line", 
        x0=0, 
        x1=0, 
        y0=0, 
        y1=1, 
        line=dict(color="gray", width=1, dash="dash"),
        xref="x", 
        yref="paper"
    )
)
st.plotly_chart(imp_fig, use_container_width=True)

# --------------------------
#   CORRELATION ANALYSIS
# --------------------------
with st.expander("Show Correlation Heatmap (Real vs. Noisy Features)"):
    st.write("Correlation matrix of all features and the target.")
    corr_matrix = df.corr()
    corr_fig = px.imshow(
        corr_matrix, 
        text_auto=False, 
        aspect="auto", 
        color_continuous_scale="RdBu_r", 
        title="Correlation Heatmap"
    )
    st.plotly_chart(corr_fig, use_container_width=True)

st.write("---")
st.write("Feel free to adjust the parameters in the sidebar and observe how these plots and metrics change!")
