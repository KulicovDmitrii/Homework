import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List

app = FastAPI(
    title="Прогноз стоимости недвижимости API",
)

# Загрузка модели и признаков при старте
try:
    model = joblib.load('house_price_model.pkl')
    model_features: List[str] = joblib.load('model_features.pkl')

    print(f"Модель загружена. Признаки: {model_features}")
except Exception as e:
    raise RuntimeError(f"Ошибка {str(e)}")


@app.get("/health", summary="health-check", tags=["health-check"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_count": len(model_features),
        "features": model_features
    }


@app.get("/predict_get", summary="Прогноз через GET", tags=["Прогноз"])
async def predict_get(
        total_square: float = Query(
            ...,
            gt=10, le=1100,
            description="Общая площадь квартиры (м²)",
        ),
        rooms: int = Query(
            ...,
            ge=1, le=6,
            description="Количество комнат",
        ),
        floor: int = Query(
            ...,
            ge=1, le=53,
            description="Этаж",
        )
):

        feature_map = {"total_square": total_square, "rooms": rooms, "floor": floor}
        input_vector = [feature_map[f] for f in model_features]
        prediction = model.predict([input_vector])[0]
        return {
            "prediction_rub": round(float(prediction), 2),
            "input": feature_map
        }



class HouseFeatures(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)







# POST
class HouseInput(BaseModel):
    total_square: float = Field(..., gt=10, le=1100, description="Площадь (м²)")
    rooms: int = Field(..., ge=1, le=6, description="Комнаты")
    floor: int = Field(..., ge=1, le=53, description="Этаж")

@app.post("/predict_post", summary="Прогноз через POST", tags=["Прогноз"])
async def predict_post(data: HouseInput):

    try:
        feature_map = data.dict()
        input_vector = [feature_map[f] for f in model_features]

        prediction = model.predict([input_vector])[0]
        return {
            "prediction_rub": round(float(prediction), 2),
            "input": feature_map
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print(f"   - Health check: http://127.0.0.1:8000/health")
    print(f"   - GET прогноз:  http://127.0.0.1:8000/predict_get?total_square=50&rooms=2&floor=3")
    print(f"   - POST прогноз: http://127.0.0.1:8000/predict_post")
    print(f"   - Документация: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")