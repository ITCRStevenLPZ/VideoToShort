from fastapi import APIRouter
from api.endpoints import process, enhanced

api_router = APIRouter()

# Include the existing process endpoints
api_router.include_router(process.router)

# Include the new enhanced endpoints
api_router.include_router(enhanced.router)