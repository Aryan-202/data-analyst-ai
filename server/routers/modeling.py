from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os

from services.model_trainer import ModelTrainer
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class ModelTrainingRequest(BaseModel):
    file_id: str
    target_column: str
    problem_type: str  # regression, classification, forecasting
    model_type: str = "auto"  # auto, specific model name
    test_size: float = 0.2
    time_column: Optional[str] = None  # for forecasting


class PredictionRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Any]]


class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_id: str
    model_info: Dict[str, Any]
    performance: Dict[str, Any]


@router.post("/train", response_model=TrainingResponse)
async def train_model(
        request: ModelTrainingRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Train machine learning model
    """
    try:
        file_manager = FileManager(settings)
        model_trainer = ModelTrainer(settings)

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Train model
        training_result = await model_trainer.train_model(
            file_path,
            request.target_column,
            request.problem_type,
            request.model_type,
            request.test_size,
            request.time_column
        )

        return TrainingResponse(
            success=True,
            message="Model training completed successfully",
            model_id=training_result['model_id'],
            model_info=training_result['model_info'],
            performance=training_result['performance']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.post("/predict")
async def make_prediction(
        request: PredictionRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Make predictions using trained model
    """
    try:
        model_trainer = ModelTrainer(settings)

        predictions = await model_trainer.predict(
            request.model_id,
            request.input_data
        )

        return {
            "success": True,
            "model_id": request.model_id,
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models/{file_id}/suggestions")
async def get_model_suggestions(
        file_id: str,
        target_column: str,
        settings: Settings = Depends(get_settings)
):
    """
    Get suggested models for the dataset
    """
    try:
        file_manager = FileManager(settings)
        model_trainer = ModelTrainer(settings)

        file_path = file_manager.get_file_path(file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        suggestions = await model_trainer.get_model_suggestions(
            file_path,
            target_column
        )

        return {
            "file_id": file_id,
            "target_column": target_column,
            "suggested_models": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))