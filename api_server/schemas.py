from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class GenerationMode(str, Enum):
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    top_k: int = Field(50, ge=0, le=200, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    stream: bool = Field(False, description="Whether to stream the response")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages", min_length=1)
    max_new_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=200)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(True)
    stream: bool = Field(False)


class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text output")
    prompt_tokens: int = Field(0, description="Number of prompt tokens")
    generated_tokens: int = Field(0, description="Number of generated tokens")
    finish_reason: str = Field("stop", description="Reason for stopping generation")


class ChatResponse(BaseModel):
    message: ChatMessage = Field(..., description="Assistant response message")
    prompt_tokens: int = Field(0)
    generated_tokens: int = Field(0)
    finish_reason: str = Field("stop")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")


class FeedbackRequest(BaseModel):
    prompt: str = Field(..., description="Original prompt")
    response: str = Field(..., description="Model response")
    rating: int = Field(..., ge=1, le=5, description="User rating 1-5")
    comment: Optional[str] = Field(None, description="Optional feedback comment")


class FeedbackResponse(BaseModel):
    status: str = Field("recorded", description="Feedback status")
    feedback_id: str = Field(..., description="Unique feedback identifier")
