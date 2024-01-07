import numpy as np
import pandas as pd
import re
from nltk.stem import LancasterStemmer
from flask import Flask, render_template, request
import pickle, json, random
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')


stemmer = LancasterStemmer()

# Define the clean_text function
def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z']", ' ', text)  # Keep only characters and single quotes
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])

    # Tokenize the words
    words = nltk.word_tokenize(text)

    # Remove stopwords using nltk stopwords set
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and len(word) > 2]

    # Join the words back into a cleaned text
    text = " ".join(words)

    # Remove repeated/leading/trailing spaces
    text = re.sub("\s[\s]+", " ", text).strip()

    return text





app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open("suggestions.json", 'r') as f:
    suggestions = json.load(f)

def analyseSentiment(resp):
    with open('vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    resp = cv.transform([resp])
    
    y_pred = model.predict(resp)
    pred = y_pred[0]
    
            

    
    result = pred
    sugg = "Genre consists of four elements or parts: character, story, plot and setting. An equation for remembering the genre is: Story (Action) + Plot + Character + Setting = Genre. This becomes an easy way to remember the elements of a genre."
    
    return result, sugg

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/result', methods=['GET', 'POST'])
def result():
    txt = request.form.get("txt")
    print(txt)
    if request.method=='POST':
        if txt!="":
            res =clean_text(txt)
            print(res)
            result, sugg = analyseSentiment(res)
            return render_template("result.html", result = result, sugg = sugg, txt=txt)
        else:
            return render_template('home.html')
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)