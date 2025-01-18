import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# App Title
st.title("Interactive Logistic Regression Demonstration")

# Sidebar for Parameters
n_real_features = st.sidebar.slider("Number of Real Features", min_value=1, max_value=10, value=1)
percent_real_features = st.sidebar.slider("Percent of Real Features", min_value=0.01, max_value=100, step=10, value=50)
n_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=1000, step=50, value=100)

# Derived Parameters
n_total_features = int(n_real_features / (percent_real_features / 100))
n_noisy_features = n_total_features - n_real_features

# Generate Data
np.random.seed(42)
y = np.random.choice([0, 1], size=n_samples)  # Binary outcome
real_features = np.column_stack([y + 0.8 * np.random.normal(size=n_samples) for _ in range(n_real_features)])
noisy_features = np.random.normal(size=(n_samples, n_noisy_features))
X = np.hstack((real_features, noisy_features))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Display Results
st.subheader("Results")
st.write(f"**Balanced Accuracy:** {balanced_acc:.3f}")
st.write(f"**Number of Real Features:** {n_real_features}")
st.write(f"**Total Features (Real + Noisy):** {n_total_features}")
st.write(f"**Number of Samples:** {n_samples}")
