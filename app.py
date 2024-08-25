import pickle
from wsgiref import simple_server
from flask import Flask, render_template, request,jsonify
from flask import Response
from flask_cors import CORS,cross_origin
import pandas as pd

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():

    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            PH = float(request.form['PH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
            data = ([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,PH,sulphates,alcohol]])
            print('data is:     ', data)

            with open("standardScalar.sav", 'rb') as f:
                scalar = pickle.load(f)

            with open("modelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)
            with open("pca_model.sav", 'rb') as f:
                pca_model = pickle.load(f)

            data_df = pd.DataFrame(data, index=[1, ])
            scaled_data = scalar.transform(data_df)
            principal_data = pca_model.transform(scaled_data)
            predict = model.predict(principal_data)

            if predict[0] == 3:
                result = 'Bad'
            elif predict[0] == 4:
                result = 'Below Average'
            elif predict[0] == 5:
                result = 'Average'
            elif predict[0] == 6:
                result = 'Good'
            elif predict[0] == 7:
                result = 'Very Good'
            else:
                result = 'Excellent'

            print('result is        ', result)
            return render_template('results.html',prediction=result);
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()