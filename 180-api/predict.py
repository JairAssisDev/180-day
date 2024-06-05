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

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return joblib.load(model_path)

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def create_lime_explainer(model, train_data, feature_names, class_names):
    return lime.lime_tabular.LimeTabularExplainer(train_data, mode="classification", 
                                                  training_labels=class_names, feature_names=feature_names, 
                                                  random_state=0)

def explain_instance(model, explainer, instance):
    instance_values = list(instance.values())  # Converter os valores da instância em uma lista
    instance_array = np.array([instance_values])  # Converter a lista em um array numpy
    predict_fn = lambda x: model.predict_proba(x)
    return explainer.explain_instance(instance_array[0], predict_fn)  # Passar apenas o array, não a lista

def extract_lime_values(lime_explanation):
    return lime_explanation.as_list()

def encode_image_to_base64(image):
    image_buffer = io.BytesIO()
    image.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    image_bytes = image_buffer.read()
    plt.close(image)  # Fechar a figura para liberar memória
    return base64.b64encode(image_bytes).decode('utf-8')

def save_image_to_base64(fig, labels=None, fig_size=(8, 6)):
    fig.set_size_inches(fig_size)  # Definir o tamanho da figura
    fig.tight_layout()  # Ajustar layout para evitar cortes
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    image_bytes = image_buffer.read()
    plt.close()

    # Codificar os bytes da imagem em base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    return {
        "image_base64": image_base64,
        "labels": labels  # Adicionamos os rótulos ao retorno da função
    }



def generate_prediction_probability_plot(prediction, probabilities):
    fig, ax = plt.subplots()
    ax.barh(["Low Risk", "High Risk"], probabilities, color=['red', 'green'], alpha=0.5)
    ax.set_xlabel('Probability')
    ax.set_title(f'Prediction: {prediction}')
    return fig

def generate_lime_explanation_plot(explanation_values):
    chart_labels = [item[0] for item in explanation_values]
    chart_values = [item[1] for item in explanation_values]
    bar_colors = ['red' if value > 0 else 'green' for value in chart_values]
    fig, ax = plt.subplots()
    ax.barh(chart_labels, chart_values, color=bar_colors)
    ax.set_xlabel('Feature Contribution')
    ax.set_ylabel('Features')
    return fig

def ver_instace(cancer_type, model_name):
    model_path = os.path.join("models", f"{cancer_type}/{model_name}.joblib")
    model = load_model(model_path)

    dataset_path = os.path.join("files", "files_dataset.csv")
    dataset = load_dataset(dataset_path)

    feature_types = {f"{i}{feature_name}": str(dataset[feature_name].dtype) 
                     for i, feature_name in enumerate(model.feature_names_in_)}

    class_names = model.classes_.tolist() if isinstance(model.classes_, np.ndarray) else model.classes_

    output_data = {
        "Feature_names": list(model.feature_names_in_),
        "Class_names": class_names,
        "Feature_types": feature_types
    }

    return json.dumps(output_data)

def cancer_predict(cancer_type, model_name, instance):
    model_path = os.path.join("models", f"{cancer_type}/{model_name}.joblib")
    model = load_model(model_path)

    feature_names = model.feature_names_in_
    instance_array = np.array([[instance[feature_name] for feature_name in feature_names]])

    prediction = model.predict(instance_array)
    probabilities = model.predict_proba(instance_array)[0] 

    dataset_path = os.path.join("files", "files_dataset.csv")
    dataset = load_dataset(dataset_path)

    train_data = dataset.loc[:,feature_names].dropna().to_numpy()
    explainer = create_lime_explainer(model, train_data, feature_names, model.classes_)
    lime_explanation = explain_instance(model, explainer, instance)

    explanation_values = extract_lime_values(lime_explanation)
    chart_labels = [item[0] for item in explanation_values]
    chart_values = [item[1] for item in explanation_values]
    bar_colors = ['red' if value > 0 else 'green' for value in chart_values]

    fig, ax = plt.subplots()
    ax.barh(chart_labels, chart_values, color=bar_colors)
    ax.set_xlabel('Feature Contribution')
    ax.set_ylabel('Features')

    image1 = save_image_to_base64(fig)

    if len(probabilities) != 2:
        raise ValueError("A lista de probabilidades deve conter exatamente duas probabilidades.")

    fig2, ax2 = plt.subplots(figsize=(10, 6))  # Definimos o tamanho da figura
    ax2.barh(["Low Risk", "High Risk"], probabilities, color=['red', 'green'], alpha=0.5)
    ax2.set_xlabel('Probabilidade')
    ax2.set_title('Predição: {}'.format(prediction[0]))
    image2 = save_image_to_base64(fig2, labels=["Low Risk", "High Risk"], fig_size=(10, 6))  # Passamos o tamanho desejado da figura

    return {
        "prediction": prediction.tolist(),
        "probabilities": probabilities.tolist(),
        "image1": image1,
        "image2": image2
    }

def diseases_and_models():
    models_dict = {}
    for cancer_type in os.listdir(MODELS_DIR):
        models_path = os.path.join(MODELS_DIR, cancer_type)
        cancer_models = [model[:-7] for model in os.listdir(models_path)]
        models_dict[cancer_type] = cancer_models

    return json.dumps(models_dict, indent=4)
