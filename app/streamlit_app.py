import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️")
st.title("Toxic Comment Classifier")
st.write("Enter a comment and check which toxic labels are detected.")

comment = st.text_area("Comment text")

if st.button("Predict"):
    if comment.strip():
        response = requests.post(API_URL, json={"text": comment})
        result = response.json()

        st.subheader("Prediction Results")
        for label, info in result["predictions"].items():
            status = "Yes" if info["predicted"] else "No"
            prob = info["probability"] * 100
            st.write(f"**{label}**: {status} — {prob:.1f}%")
    else:
        st.warning("Please enter some text.")