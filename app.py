import json
import plotly
import pandas as pd

from flask import Flask, request,render_template, url_for, jsonify
from flask_bootstrap import Bootstrap
from model import Model

from datetime import datetime
from plotly.graph_objs import Scatter

app = Flask(__name__)
Bootstrap(app)





@app.route('/')
#this links to the index page of the web app
@app.route('/index.html')
def index():


    return render_template('index.html')

#this links to the result page of the web app
@app.route('/result.html')
def predict_plot():



    #get the varaible inputs from the user
    companyname = request.args.get("companyname", "")
    ReferenceStartPeriod = request.args.get("ReferenceStartPeriod", "")
    ReferenceEndPeriod = request.args.get("ReferenceEndPeriod", "")
    PredictionDate = request.args.get("PredictionDate", "")

    
    stock_symbol = companyname.upper() #["WIKI/AMZN"]
    start_date = ReferenceStartPeriod #datetime(2017, 1, 1)
    end_date = ReferenceEndPeriod #datetime(2017, 12, 31)
    prediction_date = PredictionDate


    #build model
    arima = Model()

    #extract data from api
    arima.extract_data(stock_symbol, start_date, end_date)

    #train the data 
    arima.model_train()

    #Predict the stock price for a given date
    stock_predict = round(arima.predict(prediction_date)[1],2)
    
    #get the plot data 
    graph_data = arima.plot_data()


    #ids = ["graph-{}".format(i) for i, _ in enumerate(graph_data)]
    graphJSON = json.dumps(graph_data, cls = plotly.utils.PlotlyJSONEncoder)
    

    return render_template('result.html', stock_predict = stock_predict, graphJSON = graphJSON, prediction_date = prediction_date, stock_symbol = stock_symbol)#, ids =ids )

def main():
    app.run(debug =True)

if __name__ == '__main__':
    main()
    
    