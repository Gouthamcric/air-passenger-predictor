from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
global model,graph
import tensorflow as tf
# load the pre-trained Keras model

graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ind')
def index2():
    return render_template('index2.html')

@app.route('/predict',methods= ['POST'])
def predict():
    global graph
    with graph.as_default():
     model=load_model("models/passenger_predict.h5")
     p=request.get_data()
     from sklearn.preprocessing import MinMaxScaler
     import pandas as pd
     sc = MinMaxScaler()
     data=pd.read_csv("../final_dataset/air_passenger.csv")
     dataset_train=data
     train_set=dataset_train.iloc[:,1:2]
     train_set=train_set.append({'#Passengers':p},ignore_index=True)
     scaled_training=sc.fit_transform(train_set)
     y_predict = model.predict([[np.array([[scaled_training[144][0]]])]])
     y_pred=sc.inverse_transform(y_predict)[0][0]
    return str(y_pred)

if __name__ == '__main__':
    app.run(debug=True)
