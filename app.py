
from flask import Flask, render_template, request, redirect
# import pickle
import numpy as np
import joblib
import sys
import os

app = Flask(__name__)

# Load Model
try:
    # model= pickle.load(open('./models/medical.pkl','rb'))
    model = joblib.load('./models/medical.sav')
except:
    sys.exit('Unable to load the model')

image_url = os.path.join(os.path.join(
    'static', 'images'), 'machine-learning.jpg')


@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html', image_url=image_url)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if (request.method == "GET"):
        return redirect("/")

    sex = request.form.get('sex')
    smoker = request.form.get('smoker')
    age = request.form.get('age')
    bmi = request.form.get('bmi')
    children = request.form.get('children')
    region = request.form.get('region')

    # result = model.predict(np.array([[33, 3, 22.705, 0, 3, 4]]))
    # return [int(age), int(sex), float(bmi), int(children), int(smoker), int(region)]
    result = model.predict(np.array(
        [[int(age), int(sex), float(bmi), int(children), int(smoker), int(region)]]))

    return render_template('predict.html', result=round(result[0][0], 2), image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
