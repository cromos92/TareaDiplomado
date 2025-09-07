.PHONY: help setup dev test lint format clean migrate collectstatic install-deps start-services stop-services

# Default target
help:
	@echo "🏥 Farmacia IA - Comandos disponibles:"
	@echo ""
	@echo "  setup          - Configuración inicial completa del proyecto"
	@echo "  dev            - Iniciar servidores de desarrollo"
	@echo "  test           - Ejecutar suite completa de tests"
	@echo "  lint           - Ejecutar linting (ruff)"
	@echo "  format         - Formatear código (black + ruff)"
	@echo "  clean          - Limpiar archivos temporales"
	@echo ""
	@echo "  install-deps   - Instalar dependencias"
	@echo "  migrate        - Ejecutar migraciones de Django"
	@echo "  collectstatic  - Recopilar archivos estáticos"
	@echo "  start-services - Iniciar servicios externos (Redis, Qdrant)"
	@echo "  stop-services  - Detener servicios externos"
	@echo ""

# Setup completo del proyecto
setup: install-deps migrate collectstatic
	@echo "✅ Setup completo finalizado"
	@echo "💡 Ejecuta 'make dev' para iniciar el desarrollo"

# Instalar dependencias
install-deps:
	@echo "📦 Instalando dependencias..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry install; \
	else \
		pip install -r requirements.txt; \
	fi

# Desarrollo - iniciar servidores
dev:
	@echo "🚀 Iniciando servidores de desarrollo..."
	@echo "Django: http://localhost:8000"
	@echo "FastAPI: http://localhost:8001"
	@echo ""
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py runserver 8000 & \
		poetry run uvicorn api.main:app --reload --port 8001; \
	else \
		python manage.py runserver 8000 & \
		uvicorn api.main:app --reload --port 8001; \
	fi

# Tests
test:
	@echo "🧪 Ejecutando tests..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -v --tb=short; \
	else \
		pytest -v --tb=short; \
	fi

# Tests específicos
test-unit:
	@echo "🧪 Ejecutando tests unitarios..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/unit/ -v; \
	else \
		pytest tests/unit/ -v; \
	fi

test-integration:
	@echo "🧪 Ejecutando tests de integración..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest tests/integration/ -v; \
	else \
		pytest tests/integration/ -v; \
	fi

test-fast:
	@echo "🧪 Ejecutando tests rápidos..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -m "not slow" -v; \
	else \
		pytest -m "not slow" -v; \
	fi

# Linting
lint:
	@echo "🔍 Ejecutando linting..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run ruff check .; \
		poetry run ruff check --select I --diff .; \
	else \
		ruff check .; \
		ruff check --select I --diff .; \
	fi

# Formateo de código
format:
	@echo "🎨 Formateando código..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run black .; \
		poetry run ruff check --select I --fix .; \
		poetry run ruff format .; \
	else \
		black .; \
		ruff check --select I --fix .; \
		ruff format .; \
	fi

# Django management commands
migrate:
	@echo "🗄️  Ejecutando migraciones..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py migrate; \
	else \
		python manage.py migrate; \
	fi

makemigrations:
	@echo "🗄️  Creando migraciones..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py makemigrations; \
	else \
		python manage.py makemigrations; \
	fi

collectstatic:
	@echo "📁 Recopilando archivos estáticos..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py collectstatic --noinput; \
	else \
		python manage.py collectstatic --noinput; \
	fi

createsuperuser:
	@echo "👤 Creando superusuario..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py createsuperuser; \
	else \
		python manage.py createsuperuser; \
	fi

# Servicios externos
start-services:
	@echo "🔧 Iniciando servicios externos..."
	@echo "Iniciando Redis..."
	@if command -v brew >/dev/null 2>&1; then \
		brew services start redis; \
	elif command -v docker >/dev/null 2>&1; then \
		docker run -d --name farmacia-redis -p 6379:6379 redis:alpine; \
	fi
	@echo "Iniciando Qdrant..."
	@if command -v docker >/dev/null 2>&1; then \
		docker run -d --name farmacia-qdrant -p 6333:6333 qdrant/qdrant; \
	fi
	@echo "✅ Servicios iniciados"

stop-services:
	@echo "🛑 Deteniendo servicios externos..."
	@if command -v brew >/dev/null 2>&1; then \
		brew services stop redis; \
	fi
	@if command -v docker >/dev/null 2>&1; then \
		docker stop farmacia-redis farmacia-qdrant 2>/dev/null || true; \
		docker rm farmacia-redis farmacia-qdrant 2>/dev/null || true; \
	fi
	@echo "✅ Servicios detenidos"

# Limpieza
clean:
	@echo "🧹 Limpiando archivos temporales..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✅ Limpieza completada"

# Docker commands
docker-build:
	@echo "🐳 Construyendo imagen Docker..."
	docker-compose build

docker-up:
	@echo "🐳 Iniciando contenedores..."
	docker-compose up -d

docker-down:
	@echo "🐳 Deteniendo contenedores..."
	docker-compose down

docker-logs:
	@echo "📋 Mostrando logs de contenedores..."
	docker-compose logs -f

# Utilidades de desarrollo
shell:
	@echo "🐚 Iniciando shell de Django..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py shell; \
	else \
		python manage.py shell; \
	fi

dbshell:
	@echo "🗄️  Iniciando shell de base de datos..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python manage.py dbshell; \
	else \
		python manage.py dbshell; \
	fi

# Actualizar dependencias
update-deps:
	@echo "📦 Actualizando dependencias..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry update; \
		poetry export -f requirements.txt --output requirements.txt --without-hashes; \
	else \
		pip install --upgrade -r requirements.txt; \
	fi

# Verificación de seguridad
security-check:
	@echo "🔒 Verificando seguridad..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run safety check; \
	else \
		safety check -r requirements.txt; \
	fi

# ============================================================================
# COMANDOS DOCKER
# ============================================================================

# Docker development
up:
	@echo "🐳 Iniciando servicios con Docker Compose..."
	docker-compose up -d
	@echo "✅ Servicios iniciados:"
	@echo "  Django: http://localhost:8000"
	@echo "  FastAPI: http://localhost:8001"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Redis: localhost:6379"
	@echo "  Qdrant: http://localhost:6333"

down:
	@echo "🛑 Deteniendo servicios..."
	docker-compose down

logs:
	@echo "📋 Mostrando logs..."
	docker-compose logs -f

build:
	@echo "🔨 Construyendo imágenes..."
	docker-compose build

rebuild:
	@echo "🔨 Reconstruyendo imágenes..."
	docker-compose build --no-cache

# Docker con perfiles específicos
up-monitoring:
	@echo "🐳 Iniciando con monitoreo..."
	docker-compose --profile monitoring up -d

up-production:
	@echo "🐳 Iniciando en modo producción..."
	docker-compose --profile production up -d

# Comandos de desarrollo con Docker
dev-docker:
	@echo "🚀 Iniciando desarrollo con Docker..."
	BUILD_TARGET=development docker-compose up --build

# Comandos de base de datos
db-migrate:
	@echo "🗄️ Ejecutando migraciones en Docker..."
	docker-compose exec django python manage.py migrate

db-shell:
	@echo "🗄️ Abriendo shell de base de datos..."
	docker-compose exec postgres psql -U farmacia_user -d farmacia_ia

db-backup:
	@echo "💾 Creando backup de base de datos..."
	docker-compose exec postgres pg_dump -U farmacia_user farmacia_ia > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@echo "📥 Restaurando base de datos..."
	@read -p "Archivo de backup: " backup_file; \
	docker-compose exec -T postgres psql -U farmacia_user -d farmacia_ia < $$backup_file

# Comandos de limpieza
clean-docker:
	@echo "🧹 Limpiando Docker..."
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-volumes:
	@echo "🧹 Limpiando volúmenes..."
	docker-compose down -v
	docker volume prune -f

# Comandos de testing con Docker
test-docker:
	@echo "🧪 Ejecutando tests en Docker..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down -v

# Comandos de deployment
deploy-staging:
	@echo "🚀 Desplegando a staging..."
	railway deploy --environment staging

deploy-production:
	@echo "🚀 Desplegando a producción..."
	railway deploy --environment production

# Status de servicios
status:
	@echo "📊 Estado de servicios:"
	@docker-compose ps

health:
	@echo "🏥 Verificando salud de servicios..."
	@curl -s http://localhost:8000/health/ | python -m json.tool || echo "❌ Django no responde"
	@curl -s http://localhost:8001/healthz | python -m json.tool || echo "❌ FastAPI no responde"

# Logs específicos
logs-django:
	@docker-compose logs -f django

logs-api:
	@docker-compose logs -f api

logs-postgres:
	@docker-compose logs -f postgres

logs-redis:
	@docker-compose logs -f redis

# Comandos de monitoreo
metrics:
	@echo "📊 Métricas disponibles en:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

# Comandos de shell
shell-django:
	@docker-compose exec django python manage.py shell

shell-api:
	@docker-compose exec api python -c "from main import app; import uvicorn"

shell-postgres:
	@docker-compose exec postgres psql -U farmacia_user -d farmacia_ia

shell-redis:
	@docker-compose exec redis redis-cli

# Comandos de desarrollo rápido
quick-start: up db-migrate
	@echo "✅ Inicio rápido completado"

full-setup: build up db-migrate
	@echo "✅ Setup completo terminado"
