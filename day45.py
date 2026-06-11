from pathlib import Path
from typing import Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse




model_file = "used_car_price_model.joblib"
app = FastAPI(
    title="Used Car Price Prediction API",
    description="FastAPI app for predicting used car selling price.",
    version="1.0.0",
)

@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).with_name("index.html"))


try:
    model = joblib.load(model_file)
except FileNotFoundError:
    model = None


class CarInput(BaseModel):
    brand: str = Field(..., min_length=1, examples=["Maruti"])
    model: str = Field(..., min_length=1, examples=["Swift"])
    vehicle_age: int = Field(..., ge=0, le=40, examples=[5])
    km_driven: int = Field(..., ge=0, le=1_000_000, examples=[45000])
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"] = "Individual"
    fuel_type: Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"] = "Petrol"
    transmission_type: Literal["Manual", "Automatic"] = "Manual"
    mileage: float = Field(..., gt=0, le=50, examples=[20.4])
    engine: int = Field(..., gt=0, le=6000, examples=[1197])
    max_power: float = Field(..., gt=0, le=1000, examples=[81.8])
    seats: int = Field(..., ge=2, le=10, examples=[5])


class PredictionCreate(BaseModel):
    car: CarInput


class PredictionUpdate(BaseModel):
    car: CarInput


class PredictionResponse(BaseModel):
    id: str
    car: CarInput
    predicted_price: float


predictions_db: dict[str, PredictionResponse] = {}


def check_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model file not found. Keep used_car_price_model.joblib in the same folder as main.py.",
        )


def predict_price(car: CarInput) -> float:
    check_model_loaded()
    input_df = pd.DataFrame([car.model_dump()])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {error}",
        ) from error

    return round(float(prediction), 2)



@app.get("/")
def home():
    return {
        "message": "Used Car Price Prediction API is running",
        "docs": "/docs",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_prediction(request: PredictionCreate):
    predicted_price = predict_price(request.car)
    prediction_id = str(uuid4())

    prediction = PredictionResponse(
        id=prediction_id,
        car=request.car,
        predicted_price=predicted_price,
    )
    predictions_db[prediction_id] = prediction

    return prediction


@app.get("/predictions", response_model=list[PredictionResponse])
def get_all_predictions():
    return list(predictions_db.values())


@app.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: str):
    prediction = predictions_db.get(prediction_id)

    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction id not found",
        )

    return prediction


@app.put("/predictions/{prediction_id}", response_model=PredictionResponse)
def update_prediction(prediction_id: str, request: PredictionUpdate):
    if prediction_id not in predictions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction id not found",
        )

    predicted_price = predict_price(request.car)
    updated_prediction = PredictionResponse(
        id=prediction_id,
        car=request.car,
        predicted_price=predicted_price,
    )
    predictions_db[prediction_id] = updated_prediction

    return updated_prediction


@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: str):
    if prediction_id not in predictions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction id not found",
        )

    deleted_prediction = predictions_db.pop(prediction_id)

    return {
        "message": "Prediction deleted successfully",
        "deleted_prediction": deleted_prediction,
    }
