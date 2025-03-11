import streamlit as st 
import pickle
import sklearn
import nltk
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem import WordNetLemmatizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('lr_model.pkl', 'rb'))
nltk.download('puntk')
nltk.download('stopwords')
lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
def text_processing(text):
    remove_punch=[char for char in text if char not in string.punctuation ]
    clean_words=''.join(remove_punch)
    rsw=([word for word in clean_words.split() if word.lower() not in stop_words])
    return rsw
def lemmatize_text(words):
    # Lemmatize each word in the list
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
    # Join the lemmatized words with spaces
    return ' '.join(lemmatized_text)
st.title('Fake News Detection System')
input_sms = st.text_area('Enter the News...', height=200)
if st.button('Varify the News'):
    transform_sms = text_processing(input_sms)
    lemma=lemmatize_text(transform_sms)
    vector_input = tfidf.transform([lemma])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Real News")
    else:
        st.header('Fake News')

