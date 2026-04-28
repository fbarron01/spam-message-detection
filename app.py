import streamlit as st
import pandas as pd
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 100
FEEDBACK_FILE = "feedback_log.csv"

# Load model + tokenizer
@st.cache_resource
def load_spam_model():
    model = load_model("spam_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_spam_model()

# Prediction function
def predict_spam(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    spam_prob = float(model.predict(padded, verbose=0)[0][0])
    not_spam_prob = 1 - spam_prob

    prediction = "Spam" if spam_prob >= 0.5 else "Not Spam"

    return prediction, spam_prob, not_spam_prob

# Save feedback
def save_feedback(message, prediction, spam_prob, feedback):
    new_row = pd.DataFrame([{
        "message": message,
        "prediction": prediction,
        "spam_probability": spam_prob,
        "feedback": feedback
    }])

    if os.path.exists(FEEDBACK_FILE):
        old_data = pd.read_csv(FEEDBACK_FILE)
        updated_data = pd.concat([old_data, new_row], ignore_index=True)
    else:
        updated_data = new_row

    updated_data.to_csv(FEEDBACK_FILE, index=False)

# UI
st.title("Real-Time Spam Message Detector")
st.write("Enter a message below to check if it is spam.")

message = st.text_area("Message:", height=150)

# Prediction button
if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction, spam_prob, not_spam_prob = predict_spam(message)

        st.subheader("Result")

        if prediction == "Spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Not Spam")

        st.write(f"Spam Probability: **{spam_prob * 100:.2f}%**")
        st.write(f"Not Spam Probability: **{not_spam_prob * 100:.2f}%**")

        # Store last result
        st.session_state["last_message"] = message
        st.session_state["last_prediction"] = prediction
        st.session_state["last_spam_prob"] = spam_prob

# Feedback section
if "last_prediction" in st.session_state:
    st.subheader("Was this prediction correct?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Correct"):
            save_feedback(
                st.session_state["last_message"],
                st.session_state["last_prediction"],
                st.session_state["last_spam_prob"],
                "correct"
            )
            st.success("Feedback saved!")

    with col2:
        if st.button("Incorrect"):
            save_feedback(
                st.session_state["last_message"],
                st.session_state["last_prediction"],
                st.session_state["last_spam_prob"],
                "incorrect"
            )
            st.success("Feedback saved!")

# Optional: show feedback log
if os.path.exists(FEEDBACK_FILE):
    with st.expander("View Feedback Log"):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(feedback_df)
