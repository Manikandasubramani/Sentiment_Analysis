import pickle
import numpy as np
import tensorflow
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

#loading the transform model
tfidf = pickle.load(open('../sentiment_analysis/models/tranform.pkl','rb'))
# loading the model
clf = pickle.load(open('../sentiment_analysis/models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    inp_data = request.form['review']
    inp_data = [inp_data]
    #print(inp_data)
    vect = tfidf.transform(inp_data).toarray()
    prediction = clf.predict(vect)
    #print(prediction)
    
    output = ""
    if prediction[0] == 1:
        output = "Positive"
    else:
        output = "Negative"

    return render_template('index.html', prediction_text='The given comment is a {} comment'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
