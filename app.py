import flask
import pickle 
from flask import Flask, request, render_template


app = Flask(__name__)
modelname = pickle.load(open('LogsticRegressionModel.pkl', 'rb'))
vecname = pickle.load(open('vectorizer.pkl', 'rb'))

# HTML setup
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/words', methods=['GET','POST'])
# Process request
def formation():
    """
    Document content was sent w/POST request
    """
    if request.method == 'POST':
        data = request.form['content']
        model = pickle.load(open(modelname, 'rb'))
        vec = pickle(open(vecname, 'rb'))

        # get result
        result = getpredict(model, vec, data)[0]
        return flask.render_template('result.html', result = result)

def getpredict(model, vec, data):
    transformed_data = vec.tramsform([data])
    return model.predict(transformed_data)
