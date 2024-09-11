from pydantic import BaseModel
from typing import List

class ClipRequest(BaseModel):
    topics: List[str]
    transcript: str