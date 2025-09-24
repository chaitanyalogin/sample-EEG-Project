# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EEG Attention Classifier", layout="centered")

st.title("EEG Attention Classifier — Demo")
st.markdown("Upload a sample row (14 EEG channel values) or pick an example to predict Attentive vs Distracted.")

@st.cache_data
def load_data():
    df = pd.read_csv("eeg_eye_state_clean.csv")
    return df

@st.cache_data
def train_model(df):
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {"model": rf, "scaler": scaler, "acc": acc, "cm": cm}

# Load
st.sidebar.markdown("## Controls")
df = load_data()
st.sidebar.write(f"Data rows: {len(df)}   |  Channels: {len(df.columns)-1}")

st.info("Model will train once on startup (small dataset). Please wait a few seconds on first load.")

with st.spinner("Training model..."):
    res = train_model(df)
model = res["model"]
scaler = res["scaler"]
acc = res["acc"]
cm = res["cm"]

st.success(f"Trained RandomForest — Test accuracy: {acc:.3f}")

# Show confusion matrix
st.subheader("Confusion matrix (test set)")
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Open","Closed"], yticklabels=["Open","Closed"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Input method
st.subheader("Make a prediction")
choice = st.radio("Choose input:", ("Pick sample from dataset", "Paste 14 values (CSV row)"))

if choice == "Pick sample from dataset":
    sample_idx = st.number_input("Row index (0..N-1)", min_value=0, max_value=len(df)-1, value=0, step=1)
    row = df.drop(columns=["label"]).iloc[sample_idx].values.reshape(1,-1)
    st.write("Selected row (channels):")
    st.write(df.drop(columns=["label"]).iloc[sample_idx].to_dict())
else:
    txt = st.text_area("Paste 14 comma-separated channel values (order: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4)", height=100)
    row = None
    if txt.strip():
        try:
            vals = [float(x.strip()) for x in txt.strip().split(",")]
            if len(vals) != 14:
                st.error("Please paste exactly 14 values.")
            else:
                row = np.array(vals).reshape(1,-1)
                st.write("Parsed values:", vals)
        except Exception as e:
            st.error("Could not parse input. Ensure numeric CSV values.")

if row is not None:
    row_scaled = scaler.transform(row)
    prob = model.predict_proba(row_scaled)[0]
    pred = model.predict(row_scaled)[0]
    label_name = "Attentive (Eye Open)" if pred == 0 else "Distracted (Eye Closed)"
    st.markdown(f"### Prediction: **{label_name}**")
    st.write(f"Probability — Attentive (0): {prob[0]:.3f}  |  Distracted (1): {prob[1]:.3f}")

# show feature importance
st.subheader("Top feature importances (channels)")
imp = model.feature_importances_
channels = df.drop(columns=["label"]).columns.tolist()
imp_df = pd.DataFrame({"channel": channels, "importance": imp}).sort_values("importance", ascending=False)
st.table(imp_df.head(10).reset_index(drop=True))

st.markdown("---")
st.markdown("**Notes:** This web demo trains a RandomForest on the included public UCI EEG Eye State dataset. For real-world Raven's RPM use, collect labeled EEG during tasks and retrain the model using band-power features for better interpretability.")
