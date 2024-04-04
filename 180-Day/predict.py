import joblib
import numpy as np
import streamlit as st

model = joblib.load('SVM.joblib')

instance = np.array([3, 0,13,2,13.20,6300,0.99])


def predict_mortality(model, instance):
    return model.predict([instance])[0], model.predict_proba([instance])[0]

#prediction = model.predict([instance])
x=predict_mortality(model,instance)
print(predict_mortality(model, instance))


