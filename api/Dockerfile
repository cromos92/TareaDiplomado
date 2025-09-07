# Dockerfile para FastAPI - Farmacia IA API
FROM python:3.11-slim

# Metadatos
LABEL maintainer="Farmacia IA Team <admin@farmacia.ia>"
LABEL description="FastAPI microservice for Farmacia IA chat functionality"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Crear usuario no-root para seguridad
RUN groupadd -r fastapi && useradd -r -g fastapi fastapi

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements-api.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements-api.txt

# Copiar código de la aplicación
COPY . .

# Cambiar propietario de archivos
RUN chown -R fastapi:fastapi /app

# Cambiar a usuario no-root
USER fastapi

# Exponer puerto
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

# Comando por defecto
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
