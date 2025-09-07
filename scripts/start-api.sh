#!/bin/bash
# Script de inicio para FastAPI

set -e

echo "🚀 Iniciando Farmacia IA FastAPI..."

# Esperar a que Redis esté listo
echo "⏳ Esperando Redis..."
../scripts/wait-for.sh ${REDIS_HOST:-redis}:${REDIS_PORT:-6379} --timeout=30 -- echo "✅ Redis listo"

# Verificar configuración
echo "🔧 Verificando configuración..."
python -c "
from config import get_settings
settings = get_settings()
print(f'✅ Configuración cargada: {settings.APP_NAME}')
"

echo "✅ Inicialización completada"

# Determinar comando de ejecución
if [ "$DEBUG" = "True" ] || [ "$DEBUG" = "true" ]; then
    echo "🔧 Modo desarrollo - con reload"
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8001 \
        --reload \
        --log-level ${LOG_LEVEL:-info}
else
    echo "🚀 Modo producción - optimizado"
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8001 \
        --workers ${UVICORN_WORKERS:-4} \
        --loop uvloop \
        --http httptools \
        --log-level ${LOG_LEVEL:-info} \
        --access-log \
        --use-colors \
        --proxy-headers \
        --forwarded-allow-ips="*"
fi
