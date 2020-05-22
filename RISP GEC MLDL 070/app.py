import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import pickle
import model
# Import Heapq 
from heapq import nlargest
import numpy as np
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text=request.form['n1']
    prediction = model.text_summarizer(text)
    return render_template('index.html', prediction_text='{}'.format(prediction))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.text_summarizer(data.values())

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)