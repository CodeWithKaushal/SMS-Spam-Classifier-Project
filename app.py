import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Setting up the main title with custom styling
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Email/SMS Spam Classifier</h1>",
    unsafe_allow_html=True
)

# Developer credit
st.markdown(
    "<h4 style='text-align: center; color: #A9A9A9;'>Developed by Kaushal Divekar</h4>",
    unsafe_allow_html=True
)

# Subheader for user input section
st.subheader("Enter the message below:")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Text area for user input
    input_sms = st.text_area("Type your email or SMS message here...", height=200)

# Add a placeholder for results
result_placeholder = st.empty()

# Button to trigger prediction
if st.button('Predict'):
    with st.spinner('Analyzing...'):
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the processed message
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict the message category
        result = model.predict(vector_input)[0]

        # 4. Display the result with a color-coded header
        if result == 1:
            result_placeholder.markdown(
                "<h2 style='text-align: center; color: red;'>🚫 Spam</h2>",
                unsafe_allow_html=True
            )
        else:
            result_placeholder.markdown(
                "<h2 style='text-align: center; color: green;'>✅ Not Spam</h2>",
                unsafe_allow_html=True
            )

# Styling the sidebar and background for an enhanced UI
st.sidebar.markdown("## About")
st.sidebar.info(
    """
    This application is a simple Email/SMS Spam Classifier built using a machine learning model.
    The model analyzes your input message and predicts whether it is spam or not spam.
    """
)

# Option to provide feedback
st.sidebar.markdown("### Feedback")
st.sidebar.text_area("Provide your feedback or suggestions:")

# Footer
st.sidebar.markdown(
    "<small style='text-align: center; color: #A9A9A9;'>© 2024 Kaushal Divekar. All rights reserved.</small>",
    unsafe_allow_html=True
)
