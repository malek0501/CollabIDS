import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
from PIL import Image

# --- Modern UI Setup ---
st.set_page_config(page_title="Federated IDS Predictor", layout="wide")

# --- Background image ---
background_image_url = "cyber.jpg"  # Cybersecurity theme
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url('/home/top/ids_fl/FLIDS-main/');
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 15px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title with styling ---
st.markdown("""
    <h1 style='color: #00ffe5;'>
        ⛨️ Intrusion Detection System - Attack Predictor
    </h1>
    <p style='color: white;'>
        Upload a sample file and let the model <b>detect potential attacks</b> using a trained global model.
    </p>
""", unsafe_allow_html=True)

# --- Simulating Modal with Streamlit Expander ---
with st.expander("🔍 Explanation of the Model"):
    st.markdown(
        """
        This is a Federated Intrusion Detection System (IDS) designed to detect network attacks using machine learning models trained in a federated setting.
        
        **Federated learning** allows decentralized training across multiple devices without data transfer to a central server, ensuring privacy and data security.
        """
    )

# --- Load the global model ---
try:
    model = tf.keras.models.load_model("global_model.h5")
    with st.sidebar:
        st.title("📊 Model Information")
        st.subheader("Model Summary")
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        st.text("\n".join(summary_lines))

        if model.metrics_names:
            st.markdown("**Compiled Metrics:**")
            for metric in model.metrics_names:
                st.write(f"- {metric}")
        else:
            st.warning("No metrics compiled.")
except Exception as e:
    st.error(f"❌ Failed to load global model: {e}")
    st.stop()

# --- Upload CSV ---
st.markdown("<hr style='border-top: 1px solid #00ffe5;'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a CSV file (sample data)", type=["csv"])
run_pred = False

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if "label" in df.columns:
            y_true = df["label"]
            df = df.drop(columns=["label", "id", "attack_cat"], errors="ignore")
        else:# --- Background image ---

            y_true = None

        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = preprocessing.LabelEncoder().fit_transform(df[col])

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = preprocessing.MinMaxScaler().fit_transform(df[num_cols])

        if df.shape[1] != 42:
            st.error(f"❌ Expected 42 features, but received {df.shape[1]}. Please check your input.")
        else:
            run_pred = st.button("🔍 Run Prediction")

    except Exception as e:
        st.error(f"🚨 Error during preprocessing: {e}")

if run_pred and df is not None:
    try:
        predictions = model.predict(df)
        pred_labels = (predictions > 0.5).astype(int).flatten()

        st.subheader("🔎 Prediction Results")
        results_df = df.copy()
        results_df["Prediction"] = pred_labels
        results_df["Prediction Label"] = results_df["Prediction"].map({0: "✅ Benign", 1: "🚨 Attack"})

        def colorize(val):
            if val == "✅ Benign":
                return "background-color: #d4edda; color: black"
            elif val == "🚨 Attack":
                return "background-color: #f8d7da; color: black"
            return ""

        styled_df = results_df[["Prediction Label"]].style.applymap(colorize)
        st.dataframe(styled_df, use_container_width=True)

        benign_count = np.sum(pred_labels == 0)
        attack_count = np.sum(pred_labels == 1)
        st.toast(f"✅ Benign Samples: {benign_count}", icon='✅')
        st.toast(f"🚨 Detected Attacks: {attack_count}", icon='🔥')

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
