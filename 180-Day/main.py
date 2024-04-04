import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def oi() -> str:
    return "test-180"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8001)