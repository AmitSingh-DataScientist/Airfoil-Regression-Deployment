import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from flask import Response
import numpy as np
import pandas as pd

# starting a flask app
app = Flask(__name__)

#app = Flask(__name__, template_folder='../templates')
#app = Flask(__name__, template_folder='../templates', static_folder='../static')
#Starting with ../ moves one directory backwards and starts there.
#Starting with ../../ moves two directories backwards and starts there.

# loading the pickle file
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

# now will create an API
@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


# execution will start from here  (point of execution)
if __name__ == "__main__": 
    # app.run(debug=True) 
    app.run() #port=5004