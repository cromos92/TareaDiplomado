"""
Modelos Pydantic para la API de Farmacia IA
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
import uuid


class ChatRequest(BaseModel):
    """Modelo para requests de chat"""
    
    user_id: str = Field(
        ...,
        description="ID del usuario que envía el mensaje",
        example="user_12345"
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Texto del mensaje del usuario",
        example="¿Cuál es la dosis recomendada de paracetamol para adultos?"
    )
    location: Optional[str] = Field(
        None,
        description="Ubicación opcional del usuario (país, región)",
        example="Chile"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Valida que el texto no esté vacío después de strip"""
        if not v.strip():
            raise ValueError('El texto no puede estar vacío')
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Valida formato del user_id"""
        if not v.strip():
            raise ValueError('user_id no puede estar vacío')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "text": "¿Cuál es la dosis recomendada de paracetamol para adultos?",
                "location": "Chile"
            }
        }


class ChatResponse(BaseModel):
    """Modelo para responses de chat"""
    
    message: str = Field(
        ...,
        description="Respuesta del asistente de farmacia",
        example="La dosis recomendada de paracetamol para adultos es de 500-1000mg cada 6-8 horas, sin exceder 4g al día."
    )
    trace_id: str = Field(
        ...,
        description="ID único para rastrear la conversación",
        example="trace_abc123def456"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "La dosis recomendada de paracetamol para adultos es de 500-1000mg cada 6-8 horas, sin exceder 4g al día.",
                "trace_id": "trace_abc123def456"
            }
        }


class HealthResponse(BaseModel):
    """Modelo para response de health check"""
    
    status: str = Field(
        ...,
        description="Estado del servicio",
        example="healthy"
    )
    service: str = Field(
        ...,
        description="Nombre del servicio",
        example="farmacia-ia-api"
    )
    version: str = Field(
        ...,
        description="Versión del servicio",
        example="1.0.0"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "farmacia-ia-api",
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Modelo para responses de error"""
    
    error: str = Field(
        ...,
        description="Tipo de error",
        example="ValidationError"
    )
    detail: str = Field(
        ...,
        description="Descripción detallada del error",
        example="El texto no puede estar vacío"
    )
    trace_id: Optional[str] = Field(
        None,
        description="ID de rastreo para debugging",
        example="trace_error_123"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "El texto no puede estar vacío",
                "trace_id": "trace_error_123"
            }
        }
