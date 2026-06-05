import streamlit as st
import requests

API_URL = "https://arpitkr-toxic-comment-api.hf.space/predict"

st.set_page_config(page_title="Toxic Comment Classifier")
st.title("Toxic Comment Classifier")
st.write("Enter a comment and get toxicity predictions from the deployed API.")

text = st.text_area("Comment", placeholder="Type a comment here...")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a comment.")
    else:
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(API_URL, json={"text": text}, timeout=30)

            if response.status_code == 200:
                data = response.json()
                preds = data["predictions"]

                st.subheader("Predictions")
                for label, info in preds.items():
                    st.write(
                        f"**{label}** — predicted: `{info['predicted']}`, probability: `{info['probability']:.4f}`"
                    )
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")