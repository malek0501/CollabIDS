import streamlit as st
from time import sleep
import os
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np

from streamlit_extras.switch_page_button import switch_page
# --- Modern UI Setup ---
st.set_page_config(page_title="SecureFL - Login", layout="wide")

# --- Background image handling ---
background_image_path = "cyber.jpg"  # Replace with your image file path if local
if not os.path.isfile(background_image_path):
    st.error(f"Background image {background_image_path} not found!")

background_image_url = f"file:///{os.path.abspath(background_image_path)}"  # For local image

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 15px;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            padding-top: 60px;
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            border-radius: 10px;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
        }}
        .close:hover,
        .close:focus {{
            color: black;
            text-decoration: none;
            cursor: pointer;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title with styling ---
st.markdown("""
    <h1 style='color: #00ffe5;'>
        ⛨️ SecureFL - Future of IIoT Security
    </h1>
    <p style='color: white;'>
        Please log in to access the <b>Federated Intrusion Detection System</b>.
    </p>
""", unsafe_allow_html=True)



# --- Login Form ---
username = st.text_input("Username", placeholder="Enter your username")
password = st.text_input("Password", type="password", placeholder="Enter your password")

# --- Login Button ---
if st.button("Log In"):
    if username == "Admin" and password == "Admin":
        st.session_state.logged_in = True
        st.success("Logged in successfully!")
        sleep(0.5)
        # Add code here to switch to the next page (e.g., dashboard or other page)
        switch_page("predict")
    else:
        st.error("Incorrect username or password")



# --- Model Explanation Section ---
st.markdown("""
    <h2 style='color: #00ffe5;'>🔍 Further Information and Explanation </h2>
    <p style='color: white;'> SecureFL is a Federated Intrusion Detection System (IDS) designed to detect network attacks using machine learning models trained in a federated setting.</p>
    <p style='color: white;'><strong>Federated learning</strong> allows decentralized training across multiple devices without data transfer to a central server, ensuring privacy and data security.</p>
""", unsafe_allow_html=True)
