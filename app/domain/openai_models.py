from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import uuid

from app.core.constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


class ChatRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="请求的唯一标识符")
    messages: List[dict]
    model: str = DEFAULT_MODEL
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    top_p: Optional[float] = DEFAULT_TOP_P
    top_k: Optional[int] = DEFAULT_TOP_K
    stop: Optional[Union[List[str],str]] = None
    reasoning_effort: Optional[str] = None
    tools: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = []
    tool_choice: Optional[str] = None
    response_format: Optional[dict] = None


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-004"
    encoding_format: Optional[str] = "float"


class ImageGenerationRequest(BaseModel):
    model: str = "imagen-3.0-generate-002"
    prompt: str = ""
    n: int = 1
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = None
    style: Optional[str] = None
    response_format: Optional[str] = "url"
