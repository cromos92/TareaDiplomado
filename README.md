## Diplomado RAG 2024-S1 · Proyecto (LangChain + LangServe + Qdrant + OpenAI)

Este repo implementa un sistema RAG end‑to‑end:
- Ingesta de documentos (PDF/TXT/DOCX) a Qdrant Cloud
- Servidor LangServe con endpoints `/openai` (demo), `/rag` (RAG) y `/ingest` (ingesta web)
- RAG con retriever Qdrant, prompt optimizado y lógica "humana" para estadísticas
- Scripts de evaluación (answerable / unanswerable)
- **🆕 Múltiples interfaces web** para diferentes preferencias de usuario

### 1) Entornos y activación
- Python 3.11 (principal):
```bash
eval "$($HOME/micromamba/bin/micromamba shell hook -s zsh)"; micromamba activate rag-m3-py311
```
- Python 3.12 (secundario):
```bash
eval "$($HOME/micromamba/bin/micromamba shell hook -s zsh)"; micromamba activate rag-m3
```
- Desactivar:
```bash
micromamba deactivate
```

### 2) Variables de entorno (.env)
Crea/edita `.env` en la raíz con:
```bash
OPENAI_API_KEY=tu_openai_key
QDRANT_URL=https://<tu-endpoint>.cloud.qdrant.io
QDRANT_API_KEY=tu_qdrant_api_key
QDRANT_COLLECTION=MiColeccionRag
RAG_EMBED_MODEL=text-embedding-3-small
# Opcionales (tuning retriever)
RAG_SEARCH_TYPE=mmr
RAG_TOP_K=4
RAG_FETCH_K=20
RAG_MMR_LAMBDA=0.5
```
Si no usas gateway, evita un `OPENAI_BASE_URL` inválido; si existe, usa `https://api.openai.com/v1` o elimínala.

### 3) Ingesta a Qdrant

#### 🆕 **OPCIÓN A: Interfaz Web (RECOMENDADA)**
Accede a cualquiera de estas opciones:

- **🎨 Selector de Estilos**: `http://localhost:8000/selector` - Elige tu interfaz preferida
- **🏗️ Dashboard Completo**: `http://localhost:8000/dashboard` - Interfaz completa con cards detalladas
- **🎯 Interfaz Simple**: `http://localhost:8000/simple` - Diseño minimalista con botones grandes
- **📤 Ingesta Directa**: `http://localhost:8000/ingest/ui` - Solo para subir documentos

**Características de las interfaces:**
- **Dashboard Completo**: 6 cards con descripciones, estadísticas avanzadas, acciones rápidas
- **Interfaz Simple**: 3 botones principales grandes, diseño centrado, ideal para móviles
- **Selector**: Permite elegir entre todas las opciones disponibles

#### **OPCIÓN B: Scripts de Línea de Comandos**
Coloca documentos en `./data` (PDF, TXT, DOCX). Luego:
```bash
micromamba activate rag-m3-py311
python scripts/ingest_qdrant.py \
  --data-dir data \
  --patterns "*.pdf,*.txt,*.docx" \
  --chunker recursive \
  --embedding-model text-embedding-3-small \
  --show
```

**Presets recomendados por tipo:**
- **PDFs**: `--chunker semantic --chunk-size 1200 --chunk-overlap 200`
- **Word docs**: `--chunker semantic --chunk-size 1000 --chunk-overlap 150`  
- **Texto plano**: `--chunker recursive --chunk-size 800 --chunk-overlap 120`

### 4) Servidor LangServe (endpoints)
Levanta el server (cargando `.env`):
```bash
micromamba run -n rag-m3-py311 uvicorn app.server:app \
  --app-dir langserve-basic-example \
  --host 0.0.0.0 --port 8000 \
  --env-file .env
```

#### 🆕 **Interfaces Web Disponibles:**
- **`/selector`**: 🎨 Selector de estilos para elegir tu interfaz preferida
- **`/dashboard`**: 🏗️ Dashboard completo con todas las funcionalidades
- **`/simple`**: 🎯 Interfaz simple y minimalista
- **`/ingest/ui`**: 📤 Interfaz específica para ingesta de documentos

#### **Endpoints API:**
- **`/`**: Información general del sistema y guía de inicio rápido
- **`/openai/playground/`**: Demo de resumen de texto
- **`/rag/playground/`**: Sistema RAG completo
- **`/ingest/upload`**: API para subir documentos
- **`/ingest/status`**: Estadísticas de la colección

#### **Uso de la API de ingesta:**
```bash
# Subir documento con configuración personalizada
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@documento.pdf" \
  -F "chunker_type=semantic" \
  -F "chunk_size=1200" \
  -F "chunk_overlap=200"

# Ver estadísticas
curl http://localhost:8000/ingest/status
```

### 5) Evaluación
- Answerable:
```bash
python scripts/evaluate_rag.py \
  --answerable eval/answerable.jsonl \
  --unanswerable /dev/null \
  --report eval/report_answerable.json
```
- Unanswerable:
```bash
python scripts/evaluate_rag.py \
  --answerable /dev/null \
  --unanswerable eval/unanswerable.jsonl \
  --report eval/report_unanswerable.json
```

### 6) Comandos personalizados útiles
- Activar entorno del proyecto:
```bash
eval "$($HOME/micromamba/bin/micromamba shell hook -s zsh)"; micromamba activate rag-m3-py311
```
- Servir en 8001:
```bash
micromamba run -n rag-m3-py311 uvicorn app.server:app --app-dir langserve-basic-example --host 0.0.0.0 --port 8001 --env-file .env
```

### 7) Solución de problemas
- **Interfaz web no carga**: Verifica que los archivos HTML existan en `app/static/`
- **Error de permisos**: Asegúrate de que el directorio `app/static` sea accesible
- **Archivos no se suben**: Revisa que las variables de entorno estén configuradas
- **Dashboard no funciona**: Instala `python-multipart` con `pip install python-multipart`

### 8) Archivos clave
- **`langserve-basic-example/app/server.py`**: Servidor principal con endpoints `/openai`, `/rag`, `/ingest`, `/dashboard`, `/simple`, `/selector`
- **`langserve-basic-example/app/static/dashboard.html`**: 🏗️ Dashboard completo con cards detalladas
- **`langserve-basic-example/app/static/simple.html`**: 🎯 Interfaz simple y minimalista
- **`langserve-basic-example/app/static/selector.html`**: 🎨 Selector de estilos
- **`langserve-basic-example/app/static/index.html`**: 📤 Interfaz de ingesta de documentos
- **`scripts/ingest_qdrant.py`**: Script de ingesta por línea de comandos
- **`scripts/evaluate_rag.py`**: Evaluación por HTTP
- **`eval/answerable.jsonl`, `eval/unanswerable.jsonl`**: Datasets de evaluación

### 🚀 **Inicio Rápido**

1. **Inicia el servidor**:
```bash
cd langserve-basic-example
uvicorn app.server:app --host 0.0.0.0 --port 8000 --env-file ../.env
```

2. **Elige tu interfaz**:
   - **🎨 Selector**: `http://localhost:8000/selector` - Para elegir
   - **🏗️ Dashboard**: `http://localhost:8000/dashboard` - Completo
   - **🎯 Simple**: `http://localhost:8000/simple` - Minimalista
   - **📤 Ingesta**: `http://localhost:8000/ingest/ui` - Solo documentos

3. **¡Disfruta de tu sistema RAG con interfaz web!** 🎉