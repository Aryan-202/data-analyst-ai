from fastapi import APIRouter, HTTPException, BackgroundTasks
from server.models.dataset_schema import ModelingRequest, ModelingResponse
from server.services.model_trainer import ModelTrainer
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
model_trainer = ModelTrainer()
data_loader = DataLoader()


@router.post("/train", response_model=ModelingResponse)
async def train_model(
        background_tasks: BackgroundTasks,
        request: ModelingRequest
):
    """
    Train a machine learning model on the dataset
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in dataset"
            )

        # Train model
        model_results = await model_trainer.train_model(
            df,
            request.target_column,
            request.problem_type,
            request.model_type,
            request.test_size
        )

        # Store model (in background)
        background_tasks.add_task(
            model_trainer.store_model,
            request.dataset_id, model_results
        )

        logger.info(f"Model trained for dataset: {request.dataset_id}")

        return ModelingResponse(
            dataset_id=request.dataset_id,
            model_id=model_results.get("model_id"),
            problem_type=request.problem_type,
            target_column=request.target_column,
            model_performance=model_results.get("performance", {}),
            feature_importance=model_results.get("feature_importance"),
            predictions_sample=model_results.get("predictions_sample", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@router.get("/model-types")
async def get_available_models():
    """Get available model types for different problems"""
    return {
        "classification_models": model_trainer.get_classification_models(),
        "regression_models": model_trainer.get_regression_models(),
        "clustering_models": model_trainer.get_clustering_models(),
        "auto_ml_capabilities": model_trainer.get_auto_ml_info()
    }