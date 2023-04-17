from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData
import pandas as pd


application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def prediction_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            CRIM = float(request.form.get("CRIM"))
            , ZN=float(request.form.get("ZN"))
            , INDUS=float(request.form.get("INDUS"))
            , CHAS=int(request.form.get("CHAS"))
            , NOX=float(request.form.get("NOX"))
            , RM=float(request.form.get("RM"))
            , AGE=float(request.form.get("AGE"))
            , DIS=float(request.form.get("DIS"))
            , RAD=int(request.form.get("RAD"))
            , TAX=float(request.form.get("TAX"))
            , PTRATIO=float(request.form.get("PTRATIO"))
            , B=float(request.form.get("B"))
            , LSTAT=float(request.form.get("LSTAT"))
            )
        
        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = round(pred[0],2)

        return render_template("result.html",final_result = result)


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)


    