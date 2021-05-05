import os
import flask
from flask_cors import CORS
app = flask.Flask(__name__)
CORS(app)
from flask import Flask,jsonify, render_template,request

#load Flask 


#comment out line before production, only needed during testing:
#app.config['TESTING'] = True


#load model preprocessing
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import model_from_json
from keras.layers import Input
#

# Load pre-trained model into memory
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# Helper function for tokenizing text to feed through pre-trained deep learning network
def prepDataForDeepLearning(text):
    trainWordFeatures = tokenizer.texts_to_sequences(text)
    textTokenized = pad_sequences(trainWordFeatures, 201, padding='post')
    
    return textTokenized

# Load files needed to create proper matrix using tokens from training data
inputDataTrain = pd.DataFrame(pd.read_csv("train_DrugExp_Text.tsv", "\t", header=None))
trainText = [item[1] for item in inputDataTrain.values.tolist()]
trainingLabels = [0 if item[0] == -1 else 1 for item in inputDataTrain.values.tolist()]

VOCABULARY_SIZE=10000
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
tokenizer.fit_on_texts(trainText)

## convert words into word ids
#meanLength = np.mean([len(item.split(" ")) for item in trainText])

textTokenized = prepDataForDeepLearning(trainText)

loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# See above. The first message had a 92% probability being severe and the second had a 20% chance. 
# Appears to be working!

# define a predict function as an endpoint 
@app.route('/', methods=['GET','POST'])
def predict():
    #whenever the predict method is called, we're going
    #to input the user entered text into the model
    #and return a prediction
    if request.method=='POST':
        textData = request.form.get('text_entered')
        textDataArray = [textData]
        textTokenized = prepDataForDeepLearning(textDataArray)
        prediction = int((1-np.asscalar(loaded_model.predict(textTokenized)))*100)
        #return prediction in new page
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template("search_page.html")   


# Note: This code likely will return error message. Follow instructions below to correct error.
# You need to edit the echo function definition at ../Lib/site-packages/click/utils.py the default value for the file parameter must be sys.stdout instead of None.
# Do the same for the secho function definition at ../Lib/site-packages/click/termui.py

if __name__ == "__main__":
    # start the flask app, allow remote connections
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 8080))
    #this ensures that updates to html/css/js will come through
    app.jinja_env.auto_reload = True  
    app.run(host='127.0.0.1', port=port) 

