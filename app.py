from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"

st.set_page_config(page_title="MediPredict-AI", page_icon="🩺", layout="wide")
st.title("🩺 MediPredict-AI")
st.subheader("Educational breast-cancer classification demo")
st.warning(
    "For education only. This demo is not a medical device and must not be used "
    "to diagnose, rule out, or make treatment decisions about cancer."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. First run `python main.py` to train and save the model.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_names = saved["feature_names"]
target_names = saved["target_names"]

dataset = load_breast_cancer(as_frame=True)
reference = dataset.data

st.write(
    "The model expects 30 numeric measurements from the dataset. "
    "To keep this beginner demo simple, you can try an example record from the dataset."
)

row_number = st.slider("Choose an example record", min_value=0, max_value=len(reference) - 1, value=0)
example = reference.iloc[row_number]
input_df = pd.DataFrame([example], columns=feature_names)

with st.expander("View the 30 measurements for this example"):
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

if st.button("Run model prediction"):
    prediction = int(model.predict(input_df)[0])
    label = target_names[prediction]
    st.subheader("Model output")
    st.write(f"Predicted class for this example: **{label}**")
    st.caption(
        "This is a model prediction on a public teaching dataset, not a clinical result. "
        "Do not use it to assess a real person."
    )
