from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Create flask app
app = Flask(__name__)

@app.route('/')
def index():
    res = None
    return render_template('index.html', res = res)

@app.route('/',methods = ['GET','POST'])
def predict():
    try:
        
        sepal_length = float(request.form['s_len'])
        sepal_width = float(request.form['s_wid'])
        petal_length = float(request.form['p_len'])
        petal_width = float(request.form['p_wid'])
        if(sepal_length !=None) and (petal_length != None) and (sepal_width != None) and (petal_width!=None):
            output = 'unknown'
            model = pickle.load(open('model.pkl','rb'))
            x_input = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4)
            output=model.predict(x_input)[0]
            return render_template('index.html', res = output)
        else:
            res = 'Invalid data input'
            render_template('index.html',res = res)
    except:
        res='Unknown server error'
    return render_template('index.html',res = res)
if __name__=='__main__':
    app.run()
