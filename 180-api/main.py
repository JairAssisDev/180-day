from flask import Flask,request
from flask_cors import CORS

import predict

app = Flask(__name__)
CORS(app)

@app.route("/diseases_models", methods=["GET"])
def diseases_models():
    return predict.diseases_and_models()

@app.route("/feature_names_models", methods=['GET'])
def feature_names_models():
    doença = request.args.get('cancer_type')
    model = request.args.get('model')
    return predict.ver_instace(doença,model)

@app.route("/predict", methods=['POST'])
def cancer_predict():
    data = request.json
    type_cancer = data['type_cancer']
    model = data["model"]
    instance = data["instance"]
    return predict.cancer_predict(type_cancer,model,instance)

if __name__ == '__main__':
    app.run()
