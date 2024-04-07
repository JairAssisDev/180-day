import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import lime
import lime.lime_tabular
import pandas as pd
import json



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


def main():

    cancer_types = os.listdir(MODELS_DIR)
    print(cancer_types)

    models_dict = {}

    for cancer_type in cancer_types:
        models_path = os.path.join(MODELS_DIR, cancer_type)
        models_list = os.listdir(models_path)
        cancer_models = []
        for model in models_list:
            cancer_models.append(model[:-7])
        models_dict[cancer_type] = cancer_models
    json_data = json.dumps(models_dict, indent=4)
    print(json_data)



def ver_instace(cancer_type, model, files_dataset):
    model_path = os.path.join("models", f"{cancer_type}/{model}.joblib")
    model = joblib.load(model_path)
    feature_names, class_names = model.feature_names_in_, model.classes_

    dataset_path = os.path.join("files", f"{files_dataset}.csv")
    original_data = pd.read_csv(dataset_path)

    feature_types = {}
    for feature_name in feature_names:
        feature_type = original_data[feature_name].dtype
        feature_types[feature_name] = str(feature_type)

    class_names = class_names.tolist() if isinstance(class_names, np.ndarray) else class_names

    output_data = {
        "Feature_names": list(feature_names),
        "Class_names": class_names,
        "Feature_types": feature_types
    }

    json_output = json.dumps(output_data)

    return json_output

result_json = ver_instace("Prostate", "SVM","files_dataset")

print(result_json)

main()
