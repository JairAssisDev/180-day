from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import predict

app = FastAPI()

# Adicionando suporte para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/diseases_models")
async def diseases_models():
    return predict.diseases_and_models()

@app.get("/feature_names_models")
async def feature_names_models(cancer_type: str, model: str):
    return predict.ver_instace(cancer_type, model)

@app.post("/predict")
async def cancer_predict(data: dict):
    type_cancer = data['type_cancer']
    model = data["model"]
    instance = data["instance"]
    try:
        return predict.cancer_predict(type_cancer, model, instance)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)