# 180-day

docker build -t api180day .

docker run -d -p 8000:8000 api180day


http://127.0.0.1:8000/diseases_models

http://127.0.0.1:8000/feature_names_models?cancer_type=Prostate&model=SVM

http://127.0.0.1:8000/predict
{
  "type_cancer": "Prostate",
  "model": "SVM",
  "instance": {
    "IPAQ-SF": 3.0,
    "KATZ": 0.0,
    "KPS": 90,
    "Polypharmacy": 2.0,
    "Hemoglobin": 13.20
}
}

