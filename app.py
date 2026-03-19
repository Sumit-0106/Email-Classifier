import streamlit as st
import pickle

# load saved files
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

# input box
input_sms = st.text_area("Enter the message")

# button
if st.button("Predict"):
    
    # preprocess (if you have function)
    transformed_sms = input_sms  

    # vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam 🚨")
    else:
        st.header("Not Spam ✅")