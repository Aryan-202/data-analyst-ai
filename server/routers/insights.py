from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os

from services.insight_generator import InsightGenerator
from services.file_manager import FileManager
from config import get_settings, Settings

router = APIRouter()


class InsightRequest(BaseModel):
    file_id: str
    analysis_types: List[str] = ["trends", "correlations", "anomalies", "summary"]
    focus_areas: Optional[List[str]] = None


class ChatRequest(BaseModel):
    file_id: str
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class InsightResponse(BaseModel):
    success: bool
    message: str
    insights: Dict[str, Any]


@router.post("/generate-insights", response_model=InsightResponse)
async def generate_insights(
        request: InsightRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Generate AI-powered insights from data
    """
    try:
        file_manager = FileManager(settings)
        insight_generator = InsightGenerator(settings)

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Generate insights
        insights = await insight_generator.generate_comprehensive_insights(
            file_path,
            request.analysis_types,
            request.focus_areas
        )

        return InsightResponse(
            success=True,
            message="Insights generated successfully",
            insights=insights
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")


@router.post("/chat")
async def chat_with_data(
        request: ChatRequest,
        settings: Settings = Depends(get_settings)
):
    """
    Chat interface for asking questions about data
    """
    try:
        file_manager = FileManager(settings)
        insight_generator = InsightGenerator(settings)

        file_path = file_manager.get_file_path(request.file_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Get answer from AI
        response = await insight_generator.answer_question(
            file_path,
            request.question,
            request.conversation_history
        )

        return {
            "success": True,
            "question": request.question,
            "answer": response['answer'],
            "supporting_data": response.get('supporting_data'),
            "suggested_followups": response.get('suggested_followups', [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")