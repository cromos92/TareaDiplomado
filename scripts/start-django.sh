#!/bin/bash
# Script de inicio para Django en producción

set -e

echo "🚀 Iniciando Farmacia IA Django..."

# Esperar a que las dependencias estén listas
echo "⏳ Esperando dependencias..."
./scripts/wait-for.sh ${DATABASE_HOST:-postgres}:${DATABASE_PORT:-5432} --timeout=60 -- echo "✅ PostgreSQL listo"
./scripts/wait-for.sh ${REDIS_HOST:-redis}:${REDIS_PORT:-6379} --timeout=30 -- echo "✅ Redis listo"

# Ejecutar migraciones
echo "🗄️ Ejecutando migraciones..."
python manage.py migrate --noinput

# Crear superusuario si no existe
echo "👤 Configurando superusuario..."
python manage.py shell -c "
from django.contrib.auth.models import User
import os
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser(
        'admin', 
        os.getenv('ADMIN_EMAIL', 'admin@farmacia.ia'), 
        os.getenv('ADMIN_PASSWORD', 'changeme123')
    )
    print('✅ Superusuario creado')
else:
    print('ℹ️ Superusuario ya existe')
"

# Recopilar archivos estáticos
echo "📁 Recopilando archivos estáticos..."
python manage.py collectstatic --noinput --clear

# Verificar configuración
echo "🔧 Verificando configuración..."
python manage.py check --deploy

# Inicializar datos si es necesario
if [ "$LOAD_INITIAL_DATA" = "true" ]; then
    echo "📊 Cargando datos iniciales..."
    python manage.py loaddata initial_data.json || echo "⚠️ No se encontraron datos iniciales"
fi

# Configurar Qdrant si es necesario
if [ "$SETUP_QDRANT" = "true" ]; then
    echo "🔍 Configurando Qdrant..."
    python manage.py shell -c "
from rag.setup import setup_qdrant_collections
try:
    setup_qdrant_collections()
    print('✅ Qdrant configurado')
except Exception as e:
    print(f'⚠️ Error configurando Qdrant: {e}')
    "
fi

echo "✅ Inicialización completada"

# Determinar comando de ejecución
if [ "$DEBUG" = "True" ] || [ "$DEBUG" = "true" ]; then
    echo "🔧 Modo desarrollo - usando runserver"
    exec python manage.py runserver 0.0.0.0:8000
else
    echo "🚀 Modo producción - usando Daphne (soporta WebSockets)"
    exec daphne \
        --bind 0.0.0.0 \
        --port 8000 \
        --access-log \
        --proxy-headers \
        --verbosity ${DAPHNE_VERBOSITY:-1} \
        core.asgi:application
fi
