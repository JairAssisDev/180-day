import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import lime
import lime.lime_tabular
import pandas as pd

MODELS_DIR = "models"

def predict_mortality(model, instance):
    return model.predict([instance])[0], model.predict_proba([instance])[0]

# Lime Explanation Functions
def create_lime_explainer(model, train_data, feature_names, class_names):
    return lime.lime_tabular.LimeTabularExplainer(train_data, mode="classification", training_labels=class_names, feature_names=feature_names, random_state=0)

def explain(model, explainer, instance):
    return explainer.explain_instance(np.array(instance), model.predict_proba)

def extract_lime_values(lime_explanation):
    return lime_explanation.as_list()