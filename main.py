import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app=Flask('__name__')
model=pickle.load(open('Approval.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict_proba(final_features)
    output=prediction[0][1]*100
    print(str(prediction))
    return render_template('home.html',prediction_text="Approval chances are {}".format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    #for Direct API calls
    data=request.get_json(force=True)
    prediction=model.predict_proba([np.array(list(data.values()))])
    output=prediction[0][1]*100
    print(output)
    return jsonify(output)

if __name__=='__main__':
    app.run(debug=True)