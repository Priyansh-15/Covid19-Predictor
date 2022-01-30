import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")



@app.route("/", methods=['POST'])
def CovidPrediction():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    
    X_pred = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7]]

    X_pred = pd.DataFrame(X_pred, columns=['Body_Temp','Breath_Rate','BloodPressure','Oxygen_Level','Vaccinated','Cough_Cold','Age'])

    savename = "covidmodel.sav"

    load_model = pickle.load(open(savename, "rb"))

    predicted = load_model.predict(X_pred)

    probability = load_model.predict_proba(X_pred)[:,1][0]*100

    if predicted==1:
        output = "The patient is having Covid 19 Positive symptoms. Make sure to Quarantine Yourself from other and Stay Safe ."
        output1 = "Model Accuracy: {}".format(probability)
    else:
        output = "The patient is having Covid 19 Negative symptoms. Maintain Social Distancing and Stay Safe."
        output1 = "Model Accuracy: {}".format(100-probability)
    
    return render_template('index.html', output1=output, output2=output1, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'])