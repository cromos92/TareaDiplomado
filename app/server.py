from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from dotenv import load_dotenv
import os
import tempfile
import shutil
from pathlib import Path

# RAG imports
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Document processing imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_experimental.text_splitters import SemanticChunker as _SemanticChunker
except Exception:
    _SemanticChunker = None
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Cargar .env si existe, pero priorizar variables de entorno del sistema
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# Workaround: rebuild all Pydantic models inside langserve.validation to fix OpenAPI schema errors
def _rebuild_langserve_models():
    try:
        import langserve.validation as ls_validation  # type: ignore
        for name in dir(ls_validation):
            try:
                attr = getattr(ls_validation, name)
                # Pydantic v2: classes with model_rebuild method
                method = getattr(attr, "model_rebuild", None)
                if callable(method):
                    method(recursive=True)
            except Exception:
                # Ignore attributes that are not Pydantic models
                pass
    except Exception:
        # If validation module is not present or layout changed, skip
        pass

_rebuild_langserve_models()

summarization_assistant_template = """
You are an expert text summarization assistant with advanced analytical skills. Your task is to create comprehensive, well-structured summaries that capture the essence and key insights of the provided text.

## Instructions:
1. **Analyze the text structure** - Identify main topics, arguments, and supporting evidence
2. **Extract key information** - Focus on facts, data, conclusions, and actionable insights
3. **Maintain logical flow** - Organize summary in a coherent, logical sequence
4. **Preserve context** - Keep important context and relationships between ideas
5. **Use clear language** - Write in concise, professional language

## Output Format:
- **Main Topic:** [1-2 sentences identifying the core subject]
- **Key Points:** [3-5 bullet points with main arguments/findings]
- **Conclusions:** [1-2 sentences with main takeaways]
- **Word Count:** [Original vs Summary ratio]

## Text to Summarize:
{text_for_summarization}

## Remember:
- Be comprehensive yet concise
- Maintain accuracy and objectivity
- Highlight the most important information
- Use bullet points for clarity when appropriate
"""

summarization_assistant_prompt = PromptTemplate(
    input_variables=["text_for_summarization"],
    template=summarization_assistant_template,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
llm_chain = summarization_assistant_prompt | llm

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Summarization App",
)

# Montar archivos estáticos
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception:
    # Si no existe el directorio, crear uno temporal
    pass

# Ensure models are rebuilt right before OpenAPI generation as well
def custom_openapi():
    _rebuild_langserve_models()
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi  # type: ignore[assignment]

 
@app.get("/")
def root():
    """Endpoint raíz que redirige automáticamente al dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard", status_code=302)
# Primero, deshabilitar el OpenAPI personalizado que está causando errores
# Comentar o eliminar esta línea:
# app.openapi = custom_openapi  # type: ignore[assignment]

# Luego, limpiar completamente el dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Dashboard principal del sistema RAG."""
    try:
        with open("app/static/dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Dashboard limpio y funcional
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard RAG System</title>
            <meta charset="utf-8">
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
                .dashboard-container { max-width: 1200px; margin: 0 auto; }
                .dashboard-header { text-align: center; color: white; margin-bottom: 40px; }
                .dashboard-header h1 { font-size: 48px; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
                .dashboard-header p { font-size: 20px; opacity: 0.9; margin: 10px 0; }
                .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; }
                .card { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: transform 0.3s ease; }
                .card:hover { transform: translateY(-5px); }
                .card-header { display: flex; align-items: center; margin-bottom: 20px; }
                .card-icon { font-size: 48px; margin-right: 20px; }
                .card-title { font-size: 24px; font-weight: bold; color: #333; margin: 0; }
                .card-description { color: #666; margin-bottom: 25px; line-height: 1.6; }
                .card-button { display: inline-block; padding: 15px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 25px; font-weight: bold; transition: all 0.3s ease; }
                .card-button:hover { transform: scale(1.05); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
                .stats-preview { background: #f8f9fa; padding: 20px; border-radius: 15px; margin-top: 20px; }
                .stats-preview h4 { margin: 0 0 15px 0; color: #333; }
                .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
                .stat-item { text-align: center; }
                .stat-number { font-size: 24px; font-weight: bold; color: #667eea; }
                .stat-label { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>🚀 RAG System Dashboard</h1>
                    <p>Sistema de Retrieval-Augmented Generation - Diplomado IA 2024-S1</p>
                </div>
                
                <div class="cards-grid">
                    <!-- Card: Chat Libre con ChatGPT -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">🤖</div>
                            <h3 class="card-title">Chat Libre con ChatGPT</h3>
                        </div>
                        <p class="card-description">
                            Conversa directamente con ChatGPT sin restricciones. Pregunta lo que quieras y obtén respuestas inteligentes en tiempo real.
                        </p>
                        <a href="/chatgpt/ui" class="card-button">Iniciar Chat</a>
                    </div>
                    
                    <!-- Card: Sistema RAG -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">🔍</div>
                            <h3 class="card-title">Sistema RAG</h3>
                        </div>
                        <p class="card-description">
                            Haz preguntas sobre tus documentos. El sistema busca en tu base de conocimientos y responde basándose en la información disponible.
                        </p>
                        <a href="/rag/playground/" class="card-button">Probar RAG</a>
                    </div>
                    
                    <!-- Card: Resumen de Texto -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">📝</div>
                            <h3 class="card-title">Resumen de Texto</h3>
                        </div>
                        <p class="card-description">
                            Resúme cualquier texto usando GPT-4o. Ideal para documentos largos, artículos o cualquier contenido que necesites condensar.
                        </p>
                        <a href="/openai/playground/" class="card-button">Crear Resumen</a>
                    </div>
                    
                    <!-- Card: Subir Documentos -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">📁</div>
                            <h3 class="card-title">Subir Documentos</h3>
                        </div>
                        <p class="card-description">
                            Agrega nuevos documentos a tu base de conocimientos. Soporta PDFs, Word y archivos de texto. Los documentos se procesan automáticamente.
                        </p>
                        <a href="/ingest/ui" class="card-button">Subir Archivo</a>
                    </div>
                    
                    <!-- Card: Estadísticas del Sistema -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">📊</div>
                            <h3 class="card-title">Estadísticas del Sistema</h3>
                        </div>
                        <p class="card-description">
                            Visualiza el estado de tu sistema RAG. Ve cuántos documentos tienes, tipos de archivos y estadísticas detalladas de tu base de conocimientos.
                        </p>
                        <a href="/stats" class="card-button">Ver Estadísticas</a>
                        
                        <!-- Vista previa de estadísticas -->
                        <div class="stats-preview">
                            <h4>�� Resumen Rápido</h4>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <div class="stat-number">4</div>
                                    <div class="stat-label">Archivos</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-number">38</div>
                                    <div class="stat-label">Chunks</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-number">3</div>
                                    <div class="stat-label">Tipos</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
@app.get("/simple", response_class=HTMLResponse)
async def simple_interface():
    """Interfaz simple del sistema RAG."""
    try:
        with open("app/static/simple.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Interfaz Simple RAG System</title></head>
        <body>
            <h1>🎯 RAG System - Interfaz Simple</h1>
            <p>El archivo HTML de la interfaz simple no se encontró.</p>
            <p>Usa los endpoints directamente:</p>
            <ul>
                <li><strong>GET /ingest/status</strong> - Ver estadísticas</li>
                <li><strong>POST /ingest/upload</strong> - Subir documento</li>
                <li><strong>GET /rag/playground/</strong> - Probar RAG</li>
            </ul>
        </body>
        </html>
        """)

@app.get("/selector", response_class=HTMLResponse)
async def style_selector():
    """Selector de estilos para el dashboard."""
    try:
        with open("app/static/selector.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Selector de Estilos RAG System</title></head>
        <body>
            <h1>🎨 RAG System - Selector de Estilos</h1>
            <p>El archivo HTML del selector no se encontró.</p>
            <p>Accede directamente a:</p>
            <ul>
                <li><strong>GET /dashboard</strong> - Dashboard completo</li>
                <li><strong>GET /simple</strong> - Interfaz simple</li>
                <li><strong>GET /</strong> - Solo API</li>
            </ul>
        </body>
        </html>
        """)

@app.get("/ingest/ui", response_class=HTMLResponse)
async def ingest_ui():
    """Interfaz web para ingesta de documentos."""
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback si no existe el archivo HTML
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Ingesta de Documentos</title></head>
        <body>
            <h1>Interfaz de Ingesta</h1>
            <p>El archivo HTML no se encontró. Usa los endpoints directamente:</p>
            <ul>
                <li><strong>POST /ingest/upload</strong> - Subir documento</li>
                <li><strong>GET /ingest/status</strong> - Ver estadísticas</li>
            </ul>
        </body>
        </html>
        """)

add_routes(app, llm_chain, path="/openai")

# =============================
# RAG: retriever + prompt chain
# =============================

def _get_env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val

def _format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("file_name") or meta.get("source") or "unknown"
        page = meta.get("page")
        prefix = f"[source: {src}, page: {page}]" if page is not None else f"[source: {src}]"
        lines.append(f"{prefix}\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

def build_rag_chain() -> Any:  # returns Runnable
    # Config desde entorno
    rag_top_k = int(_get_env("RAG_TOP_K", "4") or 4)
    rag_search_type = _get_env("RAG_SEARCH_TYPE", "similarity") or "similarity"
    rag_fetch_k = int(_get_env("RAG_FETCH_K", "20") or 20)
    rag_mmr_lambda = float(_get_env("RAG_MMR_LAMBDA", "0.5") or 0.5)
    rag_embed_model = _get_env("RAG_EMBED_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
    # Qdrant
    qdrant_url = _get_env("QDRANT_URL")
    qdrant_api_key = _get_env("QDRANT_API_KEY")
    qdrant_collection = _get_env("QDRANT_COLLECTION")
    if not (qdrant_url and qdrant_api_key and qdrant_collection):
        raise RuntimeError("Faltan QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION en el entorno")

    embeddings = OpenAIEmbeddings(model=rag_embed_model)
    vectorstore = Qdrant(
        client=QdrantClient(url=qdrant_url, api_key=qdrant_api_key),
        collection_name=qdrant_collection,
        embeddings=embeddings,
    )

    search_kwargs: Dict[str, Any] = {"k": rag_top_k}
    if rag_search_type == "mmr":
        search_kwargs.update({"fetch_k": rag_fetch_k, "lambda_mult": rag_mmr_lambda})

    retriever = vectorstore.as_retriever(search_type=rag_search_type, search_kwargs=search_kwargs)

    # Prompt template
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert RAG (Retrieval-Augmented Generation) assistant. Your role is to answer questions based EXCLUSIVELY on the provided CONTEXT from the knowledge base. "
         "You must follow these strict rules:\n\n"
         "1. **ONLY use information from the CONTEXT** - Do not use external knowledge\n"
         "2. **If insufficient information exists**, respond exactly: 'No tengo información suficiente para responder esta pregunta basándome en los documentos disponibles.'\n"
         "3. **Always cite sources** - Include file_name and page number when available\n"
         "4. **Be accurate and precise** - Don't make assumptions or inferences beyond the context\n"
         "5. **Structure your response** - Use clear paragraphs and bullet points when appropriate\n"
         "6. **Maintain objectivity** - Present information factually without bias\n\n"
         "Your expertise is in analyzing and synthesizing information from the provided documents to give accurate, well-referenced answers."),
        ("human",
         "Question: {question}\n\n"
         "CONTEXT (from knowledge base):\n{context}\n\n"
         "Instructions: Answer the question using ONLY the information in the context above. "
         "If the context doesn't contain enough information, say so clearly. "
         "Always cite your sources with file names and page numbers when available.")
    ])

    llm_rag = ChatOpenAI(model="gpt-4o", temperature=0.2)

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_rag
        | StrOutputParser()
    )
    return rag_chain

rag_chain = build_rag_chain()

# =============================
# Lógica "humana": stats Qdrant
# =============================

def _compute_corpus_stats() -> Dict[str, Any]:
    qdrant_url = _get_env("QDRANT_URL")
    qdrant_api_key = _get_env("QDRANT_API_KEY")
    qdrant_collection = _get_env("QDRANT_COLLECTION")
    if not (qdrant_url and qdrant_api_key and qdrant_collection):
        return {"error": "Faltan QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION"}
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    unique_files: Set[str] = set()
    counts_by_type: Dict[str, int] = {}
    total_points = 0
    offset = None
    while True:
        points, offset = client.scroll(qdrant_collection, with_payload=True, limit=1000, offset=offset)
        if not points:
            break
        for p in points:
            total_points += 1
            payload = p.payload or {}
            meta = payload.get("metadata") or payload  # algunos VS guardan metadatos anidados
            fname = meta.get("file_name") or meta.get("source")
            if fname:
                unique_files.add(str(fname))
            dtype = meta.get("doc_type")
            if not dtype and isinstance(fname, str):
                # inferir por extensión si existe
                _, ext = os.path.splitext(fname)
                ext = (ext or "").lower()
                if ext == ".pdf":
                    dtype = "pdf"
                elif ext in {".doc", ".docx"}:
                    dtype = "word"
                elif ext in {".txt", ".md", ".rst"}:
                    dtype = "text"
            dtype = str(dtype or "unknown")
            counts_by_type[dtype] = counts_by_type.get(dtype, 0) + 1
        if offset is None:
            break
    return {
        "total_files": len(unique_files),
        "total_chunks": total_points,
        "by_type": counts_by_type,
        "samples": sorted(list(unique_files))[:10],
    }


def _maybe_answer_stats(question: str) -> Optional[str]:
    q = question.lower().strip()
    keywords = [
        "cuantos archivos", "cuántos archivos", "cuantos documentos", "cuántos documentos",
        "lista de fuentes", "listar fuentes", "cuantos chunks", "cuántos chunks",
        "how many files", "how many documents", "list sources", "how many chunks",
    ]
    if any(k in q for k in keywords):
        stats = _compute_corpus_stats()
        if "error" in stats:
            return "No puedo acceder a las estadísticas (faltan credenciales de Qdrant)."
        parts = [
            f"Total de archivos: {stats['total_files']}",
            f"Total de chunks: {stats['total_chunks']}",
            f"Por tipo: {stats['by_type']}",
        ]
        if stats.get("samples"):
            parts.append(f"Ejemplos de archivos: {stats['samples']}")
        return "\n".join(parts)
    return None


from langchain_core.runnables import RunnableLambda

def _router(input_data: Dict[str, Any]) -> str:
    question = input_data.get("question") if isinstance(input_data, dict) else str(input_data)
    direct = _maybe_answer_stats(question or "")
    if direct is not None:
        return direct
    # Delegar al RAG
    return rag_chain.invoke(question)

class RAGInput(BaseModel):
    question: str

rag_router = RunnableLambda(_router)
add_routes(app, rag_router, path="/rag", input_type=RAGInput, output_type=str)

# =============================
# Endpoint de ingesta de documentos
# =============================

def _stable_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """Genera ID estable para Qdrant usando UUID v5."""
    import uuid
    name = f"{source}|{page}|{chunk_index}|{len(content)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def _process_document(file_path: Path, chunker_type: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Procesa un documento y retorna chunks con metadatos enriquecidos."""
    docs = []
    
    try:
        # Cargar documento según tipo
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
        elif file_path.suffix.lower() in [".doc", ".docx"]:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        else:
            # Fallback para .txt, .md, .rst
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
        
        # Aplicar chunking
        if chunker_type == "semantic" and _SemanticChunker:
            try:
                chunker = _SemanticChunker(
                    embeddings=OpenAIEmbeddings(model=_get_env("RAG_EMBED_MODEL", "text-embedding-3-small")),
                    threshold_type="percentile",
                    threshold=95
                )
                chunks = chunker.split_documents(docs)
            except Exception as e:
                print(f"SemanticChunker falló, usando RecursiveCharacterTextSplitter: {e}")
                chunker = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = chunker.split_documents(docs)
        else:
            chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.split_documents(docs)
        
        # Enriquecer metadatos
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source": str(file_path),
                "file_name": file_path.name,
                "file_ext": file_path.suffix.lower(),
                "doc_type": file_path.suffix.lower().lstrip("."),
                "chunk_index": i,
                "chunker_type": chunker_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })
            # Generar ID estable
            chunk.metadata["id"] = _stable_id(
                str(file_path), 
                chunk.metadata.get("page", 0), 
                i, 
                chunk.page_content
            )
        
        return chunks
        
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return []

def _ingest_to_qdrant(docs: List[Document]) -> Dict[str, Any]:
    """Ingesta documentos a Qdrant y retorna estadísticas."""
    try:
        # Obtener configuración Qdrant
        qdrant_url = _get_env("QDRANT_URL")
        qdrant_api_key = _get_env("QDRANT_API_KEY")
        qdrant_collection = _get_env("QDRANT_COLLECTION")
        
        if not (qdrant_url and qdrant_api_key and qdrant_collection):
            raise RuntimeError("Faltan QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION")
        
        # Configurar embeddings y vectorstore
        embed_model = _get_env("RAG_EMBED_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embed_model)
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        vectorstore = Qdrant(
            client=client,
            collection_name=qdrant_collection,
            embeddings=embeddings,
        )
        
        # Ingestar documentos
        vectorstore.add_documents(docs)
        
        # Estadísticas de ingesta
        stats = {
            "success": True,
            "documents_processed": len(docs),
            "chunks_created": len(docs),
            "embedding_model": embed_model,
            "collection": qdrant_collection
        }
        
        return stats
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "documents_processed": 0
        }

@app.post("/ingest/upload")
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    chunker_type: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150)
):
    """
    Sube y ingesta un documento a Qdrant.
    
    - chunker_type: "recursive" o "semantic"
    - chunk_size: tamaño del chunk (800-1200 recomendado)
    - chunk_overlap: solapamiento entre chunks (120-200 recomendado)
    """
    
    # Validar tipo de archivo
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no soportado. Permitidos: {', '.join(allowed_extensions)}"
        )
    
    # Validar parámetros
    if chunker_type not in ["recursive", "semantic"]:
        raise HTTPException(status_code=400, detail="chunker_type debe ser 'recursive' o 'semantic'")
    
    if chunk_size < 100 or chunk_size > 5000:
        raise HTTPException(status_code=400, detail="chunk_size debe estar entre 100 y 5000")
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap debe ser >= 0 y < chunk_size")
    
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)
        
        # Procesar documento
        docs = _process_document(temp_path, chunker_type, chunk_size, chunk_overlap)
        
        if not docs:
            raise HTTPException(status_code=500, detail="No se pudieron procesar chunks del documento")
        
        # Ingestar a Qdrant
        result = _ingest_to_qdrant(docs)
        
        # Limpiar archivo temporal
        temp_path.unlink()
        
        if result["success"]:
            return JSONResponse({
                "message": "Documento ingerido exitosamente",
                "filename": file.filename,
                "stats": result
            })
        else:
            raise HTTPException(status_code=500, detail=f"Error en ingesta: {result['error']}")
            
    except Exception as e:
        # Limpiar en caso de error
        if 'temp_path' in locals():
            try:
                temp_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

@app.get("/ingest/status")
async def get_ingest_status():
    """Retorna el estado actual de la colección Qdrant."""
    try:
        stats = _compute_corpus_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return JSONResponse({
            "status": "ok",
            "collection_stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# Agregar después de la función dashboard()

# Nuevo endpoint para chat libre con ChatGPT
@app.post("/chatgpt/chat")
async def chat_with_gpt(request: dict):
    """Chat directo con ChatGPT sin restricciones."""
    try:
        message = request.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Mensaje requerido")
        
        # Usar el LLM existente para chat libre
        response = llm.invoke(message)
        
        return JSONResponse({
            "status": "success",
            "response": response.content,
            "model": "gpt-4o",
            "timestamp": str(datetime.now())
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")

# Nueva interfaz web para chat libre
@app.get("/chatgpt/ui", response_class=HTMLResponse)
async def chatgpt_ui():
    """Interfaz web para chat libre con ChatGPT."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Libre con ChatGPT</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .chat-container { max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }
            .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
            .chat-header h1 { margin: 0; font-size: 24px; }
            .chat-messages { height: 400px; overflow-y: auto; padding: 20px; background: #f8f9fa; }
            .message { margin-bottom: 15px; padding: 12px 16px; border-radius: 20px; max-width: 70%; }
            .user-message { background: #007bff; color: white; margin-left: auto; }
            .bot-message { background: #e9ecef; color: #333; }
            .chat-input { padding: 20px; background: white; border-top: 1px solid #dee2e6; }
            .input-group { display: flex; gap: 10px; }
            .chat-input input { flex: 1; padding: 12px; border: 2px solid #dee2e6; border-radius: 25px; font-size: 16px; }
            .chat-input button { padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; }
            .chat-input button:hover { opacity: 0.9; }
            .back-link { text-align: center; margin-top: 20px; }
            .back-link a { color: white; text-decoration: none; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 Chat Libre con ChatGPT</h1>
                <p>Pregunta lo que quieras sin restricciones</p>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">¡Hola! Soy ChatGPT. ¿En qué puedo ayudarte hoy?</div>
            </div>
            <div class="chat-input">
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="Escribe tu mensaje aquí..." onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">Enviar</button>
                </div>
            </div>
        </div>
        <div class="back-link">
            <a href="/dashboard">← Volver al Dashboard</a>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                // Agregar mensaje del usuario
                addMessage(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/chatgpt/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        addMessage(data.response, 'bot');
                    } else {
                        addMessage('Error: ' + data.detail, 'bot');
                    }
                } catch (error) {
                    addMessage('Error de conexión', 'bot');
                }
            }
            
            function addMessage(text, sender) {
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.textContent = text;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)
@app.get("/stats", response_class=HTMLResponse)
async def stats_page():
    """Página bonita para mostrar estadísticas del sistema."""
    try:
        # Obtener estadísticas
        stats = _compute_corpus_stats()
        
        if "error" in stats:
            stats = {"total_files": 0, "total_chunks": 0, "by_type": {}, "samples": []}
        
        stats_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Estadísticas del Sistema RAG</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
                .stats-container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }}
                .stats-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .stats-header h1 {{ margin: 0; font-size: 32px; }}
                .stats-header p {{ margin: 10px 0 0 0; font-size: 18px; opacity: 0.9; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 30px; }}
                .stat-card {{ background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 5px solid #667eea; }}
                .stat-number {{ font-size: 48px; font-weight: bold; color: #667eea; margin: 10px 0; }}
                .stat-label {{ font-size: 18px; color: #666; margin-bottom: 15px; }}
                .files-section {{ padding: 30px; background: #f8f9fa; }}
                .files-section h3 {{ color: #333; margin-bottom: 20px; }}
                .file-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .file-item {{ background: white; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }}
                .file-name {{ font-weight: bold; color: #333; }}
                .file-type {{ color: #666; font-size: 14px; }}
                .back-link {{ text-align: center; margin-top: 20px; }}
                .back-link a {{ color: white; text-decoration: none; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="stats-container">
                <div class="stats-header">
                    <h1>📊 Estadísticas del Sistema RAG</h1>
                    <p>Resumen completo de tu base de conocimientos</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total de Archivos</div>
                        <div class="stat-number">{stats.get('total_files', 0)}</div>
                        <div class="stat-description">Documentos procesados</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total de Chunks</div>
                        <div class="stat-number">{stats.get('total_chunks', 0)}</div>
                        <div class="stat-description">Fragmentos de texto</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Tipos de Archivo</div>
                        <div class="stat-number">{len(stats.get('by_type', {}))}</div>
                        <div class="stat-description">Formatos soportados</div>
                    </div>
                </div>
                
                <div class="files-section">
                    <h3>📁 Archivos en el Sistema</h3>
                    <div class="file-list">
        """
        
        # Agregar archivos individuales
        for file_name in stats.get('samples', []):
            file_type = Path(file_name).suffix.lower().lstrip('.')
            if file_type == 'pdf':
                icon = "📄"
            elif file_type in ['doc', 'docx']:
                icon = "📝"
            elif file_type == 'txt':
                icon = "📄"
            else:
                icon = "��"
                
            stats_html += f"""
                        <div class="file-item">
                            <div class="file-name">{icon} {file_name}</div>
                            <div class="file-type">Tipo: {file_type.upper()}</div>
                        </div>
            """
        
        stats_html += """
                    </div>
                </div>
            </div>
            
            <div class="back-link">
                <a href="/dashboard">← Volver al Dashboard</a>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=stats_html)
        
    except Exception as e:
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error en Estadísticas</title></head>
        <body>
            <h1>Error al cargar estadísticas</h1>
            <p>{str(e)}</p>
            <a href="/dashboard">Volver al Dashboard</a>
        </body>
        </html>
        """)