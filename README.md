# 🏥 Farmacia IA

Agente de inteligencia artificial para farmacia con LangGraph, RAG y integración MINSAL.

## 🏗️ Arquitectura del Monorepo

```
farmacia_ia/
├── api/                # FastAPI (servicio IA)
├── agents/             # LangGraph agents
├── rag/                # Qdrant + retrieval
├── data/               # Cliente MINSAL + loaders
├── security/           # cifrado + auditoría
├── conversations/      # manejo sesiones
├── frontend/           # Django templates + static
├── config/             # settings y railway
├── tests/              # test suite
└── docker/             # containerización
```

## 🚀 Setup Local

### Prerrequisitos

- Python 3.11+
- Poetry
- PostgreSQL
- Redis
- Node.js (para Tailwind CSS)

### Instalación

1. **Clonar y configurar entorno:**
   ```bash
   git clone <repository-url>
   cd farmacia_ia
   cp env.example .env
   # Editar .env con tus configuraciones
   ```

2. **Instalar dependencias:**
   ```bash
   # Usando Poetry (recomendado)
   poetry install
   poetry shell

   # O usando pip
   pip install -r requirements.txt
   ```

3. **Configurar base de datos:**
   ```bash
   # Crear base de datos PostgreSQL
   createdb farmacia_ia

   # Ejecutar migraciones
   make migrate
   # O manualmente:
   python manage.py migrate
   ```

4. **Configurar Redis:**
   ```bash
   # Iniciar Redis (macOS con Homebrew)
   brew services start redis

   # O con Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

5. **Configurar Qdrant (Vector Database):**
   ```bash
   # Con Docker
   docker run -p 6333:6333 qdrant/qdrant

   # O instalar localmente
   # Ver: https://qdrant.tech/documentation/quick-start/
   ```

### Desarrollo

```bash
# Iniciar servidor de desarrollo
make dev

# O manualmente:
python manage.py runserver 8000  # Django
uvicorn api.main:app --reload --port 8001  # FastAPI
```

### Tailwind CSS

#### Opción 1: CDN (Temporal)
Añadir en templates base:
```html
<script src="https://cdn.tailwindcss.com"></script>
```

#### Opción 2: Build Local
```bash
# Instalar dependencias
npm install -D tailwindcss @tailwindcss/forms @tailwindcss/typography
npm install -D postcss postcss-cli autoprefixer

# Configurar Tailwind
npx tailwindcss init -p

# Build CSS
npm run build:css
# O con watch mode
npm run dev:css
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
make test

# Tests específicos
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Excluir tests lentos
```

## 🔧 Comandos Útiles

```bash
# Setup completo del proyecto
make setup

# Desarrollo
make dev

# Linting y formateo
make lint
make format

# Tests
make test

# Limpieza
make clean
```

## 📁 Estructura de Módulos

### `api/` - FastAPI Service
- Endpoints REST para el agente IA
- WebSocket para chat en tiempo real
- Middleware de autenticación y rate limiting

### `agents/` - LangGraph Agents
- Agente principal de farmacia
- Workflows de consulta y recomendación
- Integración con RAG y APIs externas

### `rag/` - Retrieval Augmented Generation
- Cliente Qdrant para vectores
- Embeddings y similarity search
- Gestión de conocimiento farmacéutico

### `data/` - Data Management
- Cliente MINSAL API
- Loaders para documentos
- Procesamiento y normalización

### `security/` - Security & Audit
- Cifrado de datos sensibles
- Auditoría de conversaciones
- Gestión de permisos

### `conversations/` - Session Management
- Manejo de sesiones de usuario
- Historial de conversaciones
- Context management

### `frontend/` - Django Frontend
- Templates y vistas
- Static files (CSS, JS)
- Dashboard administrativo

### `config/` - Configuration
- Settings de Django/FastAPI
- Configuración de Railway
- Variables de entorno

## 🔐 Variables de Entorno

Copiar `env.example` a `.env` y configurar:

- `OPENAI_API_KEY`: Clave API de OpenAI
- `DATABASE_URL`: URL de PostgreSQL
- `REDIS_URL`: URL de Redis
- `QDRANT_URL`: URL de Qdrant
- `MINSAL_API_KEY`: Clave API del MINSAL
- `SECRET_KEY`: Clave secreta de Django

## 🚢 Deployment

### Railway

```bash
# Configurar Railway CLI
railway login
railway init
railway up
```

### Docker

```bash
# Build y run
docker-compose up --build

# Solo servicios
docker-compose up postgres redis qdrant
```

## 🤝 Contribución

1. Fork del proyecto
2. Crear feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

- Documentación: [Wiki del proyecto]
- Issues: [GitHub Issues]
- Slack: [Canal del equipo]

---

**Desarrollado con ❤️ para mejorar la atención farmacéutica**
