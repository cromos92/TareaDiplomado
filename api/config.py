"""
Configuración para la FastAPI app
"""

import os
from typing import List
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación FastAPI"""
    
    # Configuración básica
    APP_NAME: str = "Farmacia IA API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",  # Django dev server
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # React dev server (si se usa)
        "https://farmacia-ia.railway.app",  # Production (ejemplo)
    ]
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW: str = "1 minute"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Base de datos (si se necesita conectar a Django DB)
    DATABASE_URL: str = "sqlite:///./db.sqlite3"
    
    # Redis (para rate limiting distribuido)
    REDIS_URL: str = "redis://localhost:6379/1"
    
    # OpenAI (para IA real)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.1
    
    # Configuración de seguridad
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Obtiene la configuración de la aplicación.
    Usa lru_cache para evitar leer el archivo .env múltiples veces.
    """
    return Settings()
