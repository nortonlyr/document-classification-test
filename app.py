import flask
import pickle 
from flask import Flask, request, render_template


app = Flask(__name__)
modelname = 'LogsticRegressionModel.pkl'
vecname = 'vectorizer.pkl'

# modelname = pickle.load(open('LogsticRegressionModel.pkl', 'rb'))
# vecname = pickle.load(open('vectorizer.pkl', 'rb'))

# HTML setup
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/words', methods=['GET','POST'])
# Process request
def form():
    """
    Document content was sent w/POST request
    """
    if request.method == 'POST':
        data = request.form['content']
        model = pickle.load(open(modelname, 'rb'))
        vec = pickle.load(open(vecname, 'rb'))

        # get result
        result = getpredict(model, vec, data)[0]
        return flask.render_template('result.html', result = result)

def getpredict(model, vec, data):
    transformed_data = vec.transform([data])
    return model.predict(transformed_data)


if __name__ == "__main__":
    app.run(debug=True)