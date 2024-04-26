import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import lime
import lime.lime_tabular
import pandas as pd
import json

import base64
import io





MODELS_DIR = "models"

def predict(model, instance):
    prediction = model.predict(instance)
    probability = model.predict_proba(instance)
    return prediction, probability




def create_lime_explainer(model, train_data, feature_names, class_names):
    return lime.lime_tabular.LimeTabularExplainer(train_data, mode="classification", training_labels=class_names, feature_names=feature_names, random_state=0)

def explain(model, explainer, instance):
    return explainer.explain_instance(np.array(instance), model.predict_proba)

def extract_lime_values(lime_explanation):
    return lime_explanation.as_list()



def diseases_and_models():
    cancer_types = os.listdir(MODELS_DIR)

    models_dict = {}

    for cancer_type in cancer_types:
        models_path = os.path.join(MODELS_DIR, cancer_type)
        models_list = os.listdir(models_path)
        cancer_models = []

        for model in models_list:
            cancer_models.append(model[:-7])
        models_dict[cancer_type] = cancer_models
    json_data = json.dumps(models_dict, indent=4)
    return json_data



def ver_instace(cancer_type, model):
    files_dataset = "files_dataset"
    model_path = os.path.join("models", f"{cancer_type}/{model}.joblib")
    model = joblib.load(model_path)
    feature_names, class_names = model.feature_names_in_, model.classes_

    dataset_path = os.path.join("files", f"{files_dataset}.csv")
    original_data = pd.read_csv(dataset_path)

    feature_types = {}
    n=0
    for feature_name in feature_names:
        feature_type = original_data[feature_name].dtype
        feature_types[f"{n}"+feature_name] = str(feature_type)
        n+=1
    n=0
    class_names = class_names.tolist() if isinstance(class_names, np.ndarray) else class_names

    output_data = {
        "Feature_names": list(feature_names),
        "Class_names": class_names,
        "Feature_types": feature_types
    }

    
    json_output = json.dumps(output_data)

    return json_output

def cancer_predict(cancer_type, model, instance):
    model_path = os.path.join("models", f"{cancer_type}/{model}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = joblib.load(model_path)

    feature_names = model.feature_names_in_

    instance_array = np.array([[instance[feature_name] for feature_name in feature_names]])

    prediction = model.predict(instance_array)
    probabilities = model.predict_proba(instance_array)[0] 


    if len(probabilities) != 2:
        raise ValueError("A lista de probabilidades deve conter exatamente duas probabilidades.")


    prediction = prediction.tolist()
    probabilities = probabilities.tolist()


    return save_prediction_probability_plot(prediction, probabilities)




def encode_image_to_base64(image):
    return base64.b64encode(image).decode('utf-8')

def save_prediction_probability_plot(prediction, probabilities):
    color1='red'
    color2='green'

    if len(probabilities) != 2:
        raise ValueError("A lista de probabilidades deve conter exatamente duas probabilidades.")

    fig, ax = plt.subplots()
    ax.barh(["Low Risk", "High Risk"], probabilities, color=[color1, color2], alpha=0.5)
    ax.set_xlabel('Probabilidade')
    ax.set_title('Predição: {}'.format(prediction))

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    image_bytes = image_buffer.read()
    plt.close()

    encoded_image = encode_image_to_base64(image_bytes)

    result = {
        "prediction": prediction,
        "probabilities": probabilities,
        "image_base64": encoded_image
    }

    return json.dumps(result)
