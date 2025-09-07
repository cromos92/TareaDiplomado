#!/bin/bash

# Script de inicio para Railway
set -e

echo "🚀 Iniciando Farmacia IA en Railway..."

# Esperar a que la base de datos esté disponible
echo "⏳ Esperando base de datos..."
python manage.py migrate --check || {
    echo "📦 Ejecutando migraciones..."
    python manage.py migrate --noinput
}

# Recopilar archivos estáticos
echo "📁 Recopilando archivos estáticos..."
python manage.py collectstatic --noinput --clear

# Crear superusuario si no existe
echo "👤 Configurando usuario administrador..."
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@farmacia.ia', 'admin123')
    print('✅ Superusuario creado: admin/admin123')
else:
    print('ℹ️  Superusuario ya existe')
"

# Iniciar servidor
echo "🌟 Iniciando servidor Django..."
if [ "$DEBUG" = "True" ]; then
    echo "🔧 Modo desarrollo"
    python manage.py runserver 0.0.0.0:$PORT
else
    echo "🚀 Modo producción con Daphne"
    daphne -b 0.0.0.0 -p $PORT core.asgi:application
fi
