import time
import random
import requests
import streamlit as st

API_URL = "https://arpitkr-toxic-comment-api.hf.space/predict"

st.set_page_config(page_title="Toxic Comment Classifier")
st.title("Toxic Comment Classifier")
st.write("Enter a comment and get toxicity predictions from the deployed API.")

text = st.text_area("Comment", placeholder="Type a comment here...")


def call_api_with_retry(text, max_retries=3, timeout=30):
    delay = 1.0

    for attempt in range(max_retries):
        response = requests.post(API_URL, json={"text": text}, timeout=timeout)

        if response.status_code == 200:
            return response

        if response.status_code == 429 and attempt < max_retries - 1:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                sleep_time = float(retry_after)
            else:
                sleep_time = delay + random.uniform(0, 0.5)

            time.sleep(sleep_time)
            delay *= 2
            continue

        return response

    return response


if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a comment.")
    else:
        try:
            with st.spinner("Analyzing..."):
                response = call_api_with_retry(text)

            if response.status_code == 200:
                data = response.json()
                preds = data["predictions"]

                st.subheader("Predictions")
                for label, info in preds.items():
                    predicted = "Yes" if info["predicted"] else "No"
                    probability = info["probability"] * 100
                    st.write(f"**{label}** — {predicted}, probability: {probability:.2f}%")

            elif response.status_code == 429:
                st.error("The API is temporarily busy or rate-limited. Please wait a moment and try again.")

            else:
                try:
                    error_json = response.json()
                    st.error(f"API error {response.status_code}: {error_json}")
                except Exception:
                    st.error(f"API error {response.status_code}. Please try again later.")

        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")