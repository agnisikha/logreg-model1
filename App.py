import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
model = pickle.load(open('model.pkl','rb')) #de-serializing

# app
app = Flask(__name__) # creating server app

# routes
@app.route('/', methods=['GET','POST']) #creating route to excecute model
def predict():
    
    # get data
    data = request.get_json(force=True)
    print(type(data))
    print(data['data'])
    
    # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    data_series = pd.Series(data['data'])

    # predictions
    result = model.predict(data_series)
    print("Result ", result[0])
    # send back to browser
    output = {'results': int(result[0])}
    
    # output = {'message' : 'Test app'}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)