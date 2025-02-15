from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("models/sales_predictor.pkl")

app = FastAPI()


class SalesRequest(BaseModel):
    features: list


@app.post("/predict")
def predict_salves(request: SalesRequest):
    prediction = model.predict(np.array([request.features]))[0]
    return {"predicted_sales": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
