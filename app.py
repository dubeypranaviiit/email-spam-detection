from pathlib import Path
import streamlit as st
import joblib

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/"models"/"logistic_regression.joblib"
st.set_page_config(page_title="Email Spam Detector",page_icon="📧",layout="centered")
st.title("📧 Email Spam Detector")
st.write("Paste an email subject and body to classify it as Spam or Ham.")
st.info("This is a research demonstration. Predictions can be wrong and do not guarantee that an email is safe.")
subject=st.text_input("Subject",placeholder="Invoice update")
body=st.text_area("Email body",height=220,placeholder="Paste the email text here")
if not MODEL_PATH.exists():
    st.error("Trained model is missing. From the project folder run: python src/prepare_data.py, then python src/train_evaluate.py")
else:
    try: model=joblib.load(MODEL_PATH)
    except Exception as exc:
        model=None; st.error(f"Could not load the trained model: {exc}")
    if st.button("Classify email",type="primary"):
        text=f"{subject.strip()}\n{body.strip()}".strip()
        if not text: st.warning("Enter a subject or email body first.")
        elif model is not None:
            try:
                label=int(model.predict([text])[0])
                st.success("Spam" if label else "Ham (not spam)")
                st.caption("Classification label only; this score is not a calibrated safety probability.")
            except Exception as exc: st.error(f"Prediction failed: {exc}")
