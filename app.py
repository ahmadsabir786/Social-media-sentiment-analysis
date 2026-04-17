import streamlit as st
import pickle

model = pickle.load(open("tfidf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("💬 Emotion Detection App")
st.write("Enter a sentence and detect emotion")

user_input = st.text_area("Type your text here:")

# 🔥 IMPORTANT: numeric label mapping (apne model ke hisaab se adjust karo)
label_map = {
    0: "sadness",
    1: "joy",
    2: "anger",
    3: "fear",
    4: "love",
    5: "surprise"
}

emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

def predict_emotion(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]   # numeric output aata hai
    emotion = label_map.get(pred, "unknown")
    return emotion

if st.button("Predict Emotion"):
    if user_input:
        result = predict_emotion(user_input)
        emoji = emojis.get(result, "")
        st.success(f"Emotion: {result} {emoji}")
    else:
        st.warning("Please enter some text!")