from flask import Flask,jsonify, request
import json
import pickle
import numpy as np




model = pickle.load(open('linear_model.pickle','rb'))

with open("columns.json", 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]


app = Flask(__name__)



@app.route('/', methods = ['POST'])

def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])


    response = jsonify({

        'estimated_price': get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)


if __name__ == '__main__':
    print(" Starting flask server")
    app.run()
