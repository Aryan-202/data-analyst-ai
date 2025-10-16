from fastapi import APIRouter, HTTPException
from server.models.dataset_schema import DataCleaningRequest, DataCleaningResponse
from server.services.data_cleaner import DataCleaner
from server.services.data_loader import DataLoader
from server.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger()
data_cleaner = DataCleaner()
data_loader = DataLoader()

@router.post("/clean", response_model=DataCleaningResponse)
async def clean_dataset(request: DataCleaningRequest):
    """
    Perform data cleaning operations on a dataset
    """
    try:
        # Load dataset
        df = data_loader.load_dataset_by_id(request.dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        rows_before = len(df)
        
        # Perform cleaning operations
        cleaned_df, operations_performed, cleaning_report = await data_cleaner.clean_dataframe(
            df, request.operations, request.options or {}
        )
        
        # Save cleaned dataset
        cleaned_dataset_id = f"cleaned_{request.dataset_id}"
        data_loader.store_dataset(cleaned_df, cleaned_dataset_id, "cleaned_dataset")
        
        rows_after = len(cleaned_df)
        
        logger.info(f"Data cleaning completed for dataset: {request.dataset_id}")
        
        return DataCleaningResponse(
            dataset_id=request.dataset_id,
            cleaned_dataset_id=cleaned_dataset_id,
            operations_performed=operations_performed,
            rows_before=rows_before,
            rows_after=rows_after,
            cleaning_report=cleaning_report
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

@router.get("/cleaning-options")
async def get_cleaning_options():
    """Get available data cleaning operations"""
    return {
        "available_operations": data_cleaner.get_available_operations(),
        "default_options": data_cleaner.get_default_options()
    }