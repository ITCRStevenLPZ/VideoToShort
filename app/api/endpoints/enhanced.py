import logging
from fastapi import APIRouter, HTTPException, Body, Depends
from models.clip_request import EnhancedClipRequest, EnhancedProcessingResponse
from services.enhanced_processor import EnhancedProcessor
from typing import List, Dict, Any

# Get a logger specific to this module
logger = logging.getLogger(__name__)

# Create a router for enhanced processing endpoints
router = APIRouter(
    prefix="/enhanced",
    tags=["enhanced"],
    responses={404: {"description": "Not found"}},
)

# Global processor instance
enhanced_processor = EnhancedProcessor()

@router.post("/process", response_model=EnhancedProcessingResponse)
async def process_video(request: EnhancedClipRequest = Body(...)):
    """
    Process a video using the enhanced flow with vector indexing and sentiment analysis.
    
    This endpoint implements the complete flowchart with:
    - Vector index creation/loading
    - Sentiment analysis
    - Customizable clip generation strategies
    
    Returns:
        EnhancedProcessingResponse with detailed clip information
    """
    try:
        logger.info(f"Processing request for video: {request.video_path}")
        response = enhanced_processor.process_video(request)
        return response
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indices", response_model=List[str])
async def list_indices():
    """
    List all available vector indices.
    
    Returns:
        List of index names
    """
    try:
        indices = enhanced_processor.get_available_indices()
        return indices
    except Exception as e:
        logger.error(f"Error listing indices: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))