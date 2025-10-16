from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from server.config import settings
from server.routers import (
    data_upload, 
    data_cleaning, 
    eda, 
    visualization, 
    insights, 
    modeling, 
    reports
)
from server.utils.logger import setup_logger

# Setup logger
logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Data Analyst AI Server")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Data Analyst AI Server")

app = FastAPI(
    title="Data Analyst AI API",
    description="AI-powered automated data analysis platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for reports
app.mount("/reports", StaticFiles(directory="data/reports"), name="reports")

# Include routers
app.include_router(data_upload.router, prefix="/api/v1", tags=["Data Upload"])
app.include_router(data_cleaning.router, prefix="/api/v1", tags=["Data Cleaning"])
app.include_router(eda.router, prefix="/api/v1", tags=["EDA"])
app.include_router(visualization.router, prefix="/api/v1", tags=["Visualization"])
app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])
app.include_router(modeling.router, prefix="/api/v1", tags=["Modeling"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

@app.get("/")
async def root():
    return {
        "message": "Data Analyst AI API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )