"""
FastAPI app principal para Farmacia IA
"""

import uuid
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

from .models import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from .config import get_settings
from .logging_config import configure_logging

# Configurar logging
configure_logging()
logger = structlog.get_logger()

# Configuración
settings = get_settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Crear app FastAPI
app = FastAPI(
    title="Farmacia IA API",
    description="""
    🏥 **API de Farmacia IA**
    
    Microservicio especializado para consultas de chat con inteligencia artificial farmacéutica.
    
    ## Características
    
    * **Chat inteligente**: Consultas sobre medicamentos, dosis, interacciones
    * **Rate limiting**: Protección contra abuso (10 requests/minuto por IP)
    * **Trazabilidad**: Cada request tiene un trace_id único
    * **Validación**: Validación estricta de inputs con Pydantic
    * **CORS**: Configurado para frontend Django
    
    ## Uso
    
    1. Envía un mensaje al endpoint `/api/chat`
    2. Recibe una respuesta del asistente de farmacia
    3. Usa el `trace_id` para rastrear la conversación
    """,
    version="1.0.0",
    contact={
        "name": "Farmacia IA Team",
        "email": "admin@farmacia.ia",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware para logging de requests"""
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    
    start_time = time.time()
    
    # Log request
    logger.info(
        "request_started",
        trace_id=trace_id,
        method=request.method,
        url=str(request.url),
        client_ip=get_remote_address(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            "request_completed",
            trace_id=trace_id,
            status_code=response.status_code,
            process_time=process_time,
        )
        
        # Agregar trace_id a headers
        response.headers["X-Trace-ID"] = trace_id
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "request_failed",
            trace_id=trace_id,
            error=str(e),
            process_time=process_time,
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler personalizado para HTTPExceptions"""
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    logger.warning(
        "http_exception",
        trace_id=trace_id,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
            trace_id=trace_id
        ).dict(),
        headers={"X-Trace-ID": trace_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para excepciones generales"""
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    logger.error(
        "internal_error",
        trace_id=trace_id,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="Ha ocurrido un error interno del servidor",
            trace_id=trace_id
        ).dict(),
        headers={"X-Trace-ID": trace_id}
    )


@app.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Verifica el estado del servicio FastAPI"
)
async def health_check():
    """
    Endpoint de health check para verificar que el servicio está funcionando.
    
    Retorna información básica del servicio incluyendo:
    - Estado actual (healthy/unhealthy)
    - Nombre del servicio
    - Versión actual
    """
    return HealthResponse(
        status="healthy",
        service="farmacia-ia-api",
        version="1.0.0"
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat con Asistente de Farmacia",
    description="Envía un mensaje al asistente de IA especializado en farmacia",
    responses={
        200: {
            "description": "Respuesta exitosa del asistente",
            "content": {
                "application/json": {
                    "example": {
                        "message": "La dosis recomendada de paracetamol para adultos es de 500-1000mg cada 6-8 horas, sin exceder 4g al día.",
                        "trace_id": "trace_abc123def456"
                    }
                }
            }
        },
        422: {
            "description": "Error de validación",
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "detail": "El texto no puede estar vacío",
                        "trace_id": "trace_error_123"
                    }
                }
            }
        },
        429: {
            "description": "Rate limit excedido",
            "content": {
                "application/json": {
                    "example": {
                        "error": "RateLimitExceeded",
                        "detail": "Demasiadas requests. Límite: 10 por minuto",
                        "trace_id": "trace_rate_limit_456"
                    }
                }
            }
        }
    }
)
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest
) -> ChatResponse:
    """
    Procesa un mensaje de chat y retorna la respuesta del asistente de farmacia.
    
    **Rate Limiting**: 10 requests por minuto por IP
    
    **Parámetros**:
    - **user_id**: Identificador del usuario
    - **text**: Mensaje del usuario (1-10000 caracteres)
    - **location**: Ubicación opcional del usuario
    
    **Respuesta**:
    - **message**: Respuesta del asistente
    - **trace_id**: ID único para rastrear la conversación
    """
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    logger.info(
        "chat_request_received",
        trace_id=trace_id,
        user_id=chat_request.user_id,
        text_length=len(chat_request.text),
        location=chat_request.location,
    )
    
    try:
        # TODO: Aquí iría la integración con el agente de IA
        # Por ahora, respuesta simulada basada en el contenido
        
        text_lower = chat_request.text.lower()
        
        if any(word in text_lower for word in ['paracetamol', 'acetaminofén']):
            response_message = """
            **Paracetamol (Acetaminofén)**
            
            📋 **Dosis para adultos**: 500-1000mg cada 6-8 horas
            ⚠️ **Dosis máxima**: No exceder 4g (4000mg) en 24 horas
            🕒 **Duración**: Máximo 3 días para fiebre, 10 días para dolor
            
            **Precauciones**:
            - No combinar con otros medicamentos que contengan paracetamol
            - Consultar médico si tienes problemas hepáticos
            - Evitar alcohol durante el tratamiento
            
            ⚕️ *Esta información es orientativa. Consulta siempre con un profesional de la salud.*
            """.strip()
            
        elif any(word in text_lower for word in ['ibuprofeno', 'advil', 'nurofen']):
            response_message = """
            **Ibuprofeno**
            
            📋 **Dosis para adultos**: 400-600mg cada 6-8 horas
            ⚠️ **Dosis máxima**: No exceder 2400mg en 24 horas
            🍽️ **Administración**: Tomar con alimentos para proteger el estómago
            
            **Contraindicaciones**:
            - Úlceras gástricas activas
            - Problemas renales graves
            - Alergia a AINEs
            
            ⚕️ *Esta información es orientativa. Consulta siempre con un profesional de la salud.*
            """.strip()
            
        elif any(word in text_lower for word in ['aspirina', 'ácido acetilsalicílico']):
            response_message = """
            **Aspirina (Ácido Acetilsalicílico)**
            
            📋 **Dosis para dolor/fiebre**: 500-1000mg cada 4-6 horas
            📋 **Dosis cardioprotectora**: 75-100mg diarios
            ⚠️ **Dosis máxima**: No exceder 4g en 24 horas
            
            **Advertencias importantes**:
            - No usar en menores de 16 años (Síndrome de Reye)
            - Puede causar sangrado gástrico
            - Interactúa con anticoagulantes
            
            ⚕️ *Esta información es orientativa. Consulta siempre con un profesional de la salud.*
            """.strip()
            
        elif any(word in text_lower for word in ['interacción', 'interacciones', 'combinar']):
            response_message = """
            **Interacciones Medicamentosas**
            
            🔍 Para consultar interacciones específicas, necesito saber:
            - ¿Qué medicamentos quieres combinar?
            - ¿Tienes alguna condición médica especial?
            
            **Reglas generales**:
            - Nunca combines medicamentos sin consultar
            - Lee siempre los prospectos
            - Informa a tu médico sobre todos los medicamentos que tomas
            
            **Interacciones comunes a evitar**:
            - Alcohol + Paracetamol (daño hepático)
            - Aspirina + Anticoagulantes (riesgo de sangrado)
            - AINEs + Corticoides (úlceras gástricas)
            
            ⚕️ *Para consultas específicas, contacta a tu farmacéutico o médico.*
            """.strip()
            
        else:
            response_message = """
            Hola, soy tu asistente de farmacia especializado en medicamentos y salud.
            
            Puedo ayudarte con:
            🔹 Información sobre medicamentos
            🔹 Dosis y administración
            🔹 Efectos secundarios
            🔹 Interacciones medicamentosas
            🔹 Consejos de salud general
            
            ¿En qué puedo ayudarte específicamente?
            
            ⚕️ *Recuerda que esta información es orientativa y no reemplaza la consulta médica profesional.*
            """
        
        logger.info(
            "chat_response_generated",
            trace_id=trace_id,
            response_length=len(response_message),
        )
        
        return ChatResponse(
            message=response_message,
            trace_id=trace_id
        )
        
    except Exception as e:
        logger.error(
            "chat_processing_error",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Error procesando la consulta de chat"
        )


# Endpoint adicional para métricas (opcional)
@app.get(
    "/metrics",
    tags=["Health"],
    summary="Métricas del servicio",
    description="Información básica de métricas y estadísticas"
)
async def get_metrics():
    """
    Retorna métricas básicas del servicio.
    Útil para monitoreo y observabilidad.
    """
    return {
        "service": "farmacia-ia-api",
        "version": "1.0.0",
        "uptime": "N/A",  # TODO: Implementar cálculo de uptime
        "requests_total": "N/A",  # TODO: Implementar contador
        "active_connections": "N/A"  # TODO: Implementar contador
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
