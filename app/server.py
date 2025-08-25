from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

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

# Cargar variables desde el .env más cercano (prioriza el de la raíz del proyecto)
# sin sobreescribir variables ya presentes en el entorno
load_dotenv(find_dotenv(usecwd=True) or None, override=False)

# Directorio absoluto para archivos estáticos (independiente del CWD)
STATIC_DIR = (Path(__file__).resolve().parent / "static")

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
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception:
    # Si no existe el directorio, ignorar
    pass

@app.get("/")
def root():
    """Endpoint raíz que redirige automáticamente al dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard", status_code=302)


# Luego, limpiar completamente el dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Dashboard principal del sistema RAG."""
    try:
        with open(STATIC_DIR / "dashboard.html", "r", encoding="utf-8") as f:
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
        with open(STATIC_DIR / "simple.html", "r", encoding="utf-8") as f:
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
        with open(STATIC_DIR / "selector.html", "r", encoding="utf-8") as f:
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
        with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
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

@app.get("/ingest/playground/")
async def ingest_playground():
    """Simple playground interface for document ingestion."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Ingestion Playground</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; }
            input, select { width: 100%; padding: 8px; margin: 5px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Document Ingestion Playground</h1>
            <p>Upload and process documents:</p>
            <form id="uploadForm">
                <div class="form-group">
                    <label for="file">Select File:</label>
                    <input type="file" id="file" name="file" accept=".pdf,.docx,.doc,.txt,.md,.rst" required>
                </div>
                <div class="form-group">
                    <label for="chunkerType">Chunker Type:</label>
                    <select id="chunkerType" name="chunker_type">
                        <option value="recursive">Recursive</option>
                        <option value="semantic">Semantic</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="chunkSize">Chunk Size:</label>
                    <input type="number" id="chunkSize" name="chunk_size" value="1000" min="100" max="5000">
                </div>
                <div class="form-group">
                    <label for="chunkOverlap">Chunk Overlap:</label>
                    <input type="number" id="chunkOverlap" name="chunk_overlap" value="150" min="0" max="1000">
                </div>
                <button type="submit">Upload Document</button>
            </form>
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const chunkerType = document.getElementById('chunkerType').value;
                const chunkSize = document.getElementById('chunkSize').value;
                const chunkOverlap = document.getElementById('chunkOverlap').value;
                
                if (!fileInput.files[0]) {
                    alert('Please select a file');
                    return;
                }
                
                formData.append('file', fileInput.files[0]);
                formData.append('chunker_type', chunkerType);
                formData.append('chunk_size', chunkSize);
                formData.append('chunk_overlap', chunkOverlap);
                
                try {
                    const response = await fetch('/ingest/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('result').innerHTML = '<h3>Success:</h3><p>' + data.message + '</p>';
                        document.getElementById('result').style.display = 'block';
                    } else {
                        document.getElementById('result').innerHTML = '<h3>Error:</h3><p>' + data.detail + '</p>';
                        document.getElementById('result').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<h3>Error:</h3><p>Network error occurred</p>';
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """)

# Simple OpenAI endpoints
@app.post("/openai/summarize")
async def summarize_text(request: dict):
    """Summarize text using the LLM chain."""
    try:
        text = request.get("text_for_summarization", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        response = llm.invoke(text)
        return {"summary": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai/playground/")
async def openai_playground():
    """Interfaz moderna para resumen de texto con estilo chileno."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📝 Resumen de Texto - IA Chile</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .chilean-header {
                background: linear-gradient(135deg, #d52b1e 0%, #f4a460 50%, #ffffff 100%);
                padding: 30px 0;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                position: relative;
                overflow: hidden;
            }
            
            .chilean-header::before {
                content: '🇨🇱';
                font-size: 48px;
                position: absolute;
                top: 20px;
                left: 30px;
                animation: wave 2s ease-in-out infinite;
            }
            
            .chilean-header::after {
                content: '🇨🇱';
                font-size: 48px;
                position: absolute;
                top: 20px;
                right: 30px;
                animation: wave 2s ease-in-out infinite 1s;
            }
            
            @keyframes wave {
                0%, 100% { transform: rotate(0deg); }
                25% { transform: rotate(20deg); }
                75% { transform: rotate(-20deg); }
            }
            
            .header-content h1 {
                font-size: 3.5rem;
                color: #1e3c72;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                font-weight: 800;
            }
            
            .header-content p {
                font-size: 1.2rem;
                color: #2a5298;
                font-weight: 500;
            }
            
            .main-container {
                max-width: 1200px;
                margin: 40px auto;
                padding: 0 20px;
            }
            
            .playground-card {
                background: white;
                border-radius: 25px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
                margin-bottom: 30px;
            }
            
            .input-section {
                padding: 40px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }
            
            .input-section h2 {
                color: #1e3c72;
                font-size: 2rem;
                margin-bottom: 20px;
                text-align: center;
                font-weight: 700;
            }
            
            .input-section p {
                color: #6c757d;
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.1rem;
            }
            
            .textarea-container {
                position: relative;
                margin-bottom: 30px;
            }
            
            .textarea-container textarea {
                width: 100%;
                height: 250px;
                padding: 25px;
                border: 3px solid #e9ecef;
                border-radius: 20px;
                font-size: 16px;
                line-height: 1.6;
                resize: vertical;
                transition: all 0.3s ease;
                font-family: 'Segoe UI', sans-serif;
                background: white;
            }
            
            .textarea-container textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 6px rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }
            
            .textarea-container textarea::placeholder {
                color: #adb5bd;
                font-style: italic;
            }
            
            .button-group {
                display: flex;
                gap: 20px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 18px 40px;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                min-width: 200px;
            }
            
            .btn-primary:hover {
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                border: none;
                padding: 18px 40px;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 8px 25px rgba(108, 117, 125, 0.3);
                min-width: 200px;
            }
            
            .btn-secondary:hover {
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(108, 117, 125, 0.4);
            }
            
            .output-section {
                padding: 40px;
                background: white;
                display: none;
            }
            
            .output-section h3 {
                color: #1e3c72;
                font-size: 1.8rem;
                margin-bottom: 20px;
                text-align: center;
                font-weight: 700;
            }
            
            .summary-content {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 30px;
                border-radius: 20px;
                border-left: 6px solid #667eea;
                font-size: 16px;
                line-height: 1.8;
                color: #333;
                white-space: pre-wrap;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #667eea;
                font-size: 18px;
            }
            
            .loading::after {
                content: '';
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #667eea;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 1s linear infinite;
                margin-left: 10px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .stats-section {
                background: white;
                padding: 40px;
                border-top: 1px solid #e9ecef;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .stat-item {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 15px;
                border-left: 4px solid #667eea;
            }
            
            .stat-number {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .stat-label {
                color: #6c757d;
                font-size: 0.9rem;
            }
            
            .back-link {
                text-align: center;
                margin-top: 30px;
            }
            
            .back-link a {
                display: inline-block;
                background: linear-gradient(135deg, #d52b1e 0%, #f4a460 100%);
                color: white;
                text-decoration: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(213, 43, 30, 0.3);
            }
            
            .back-link a:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(213, 43, 30, 0.4);
            }
            
            .chilean-footer {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                text-align: center;
                padding: 20px;
                margin-top: 40px;
                border-radius: 25px 25px 0 0;
            }
            
            .chilean-footer p {
                font-size: 14px;
                opacity: 0.8;
            }
            
            @media (max-width: 768px) {
                .header-content h1 { font-size: 2.5rem; }
                .input-section, .output-section { padding: 20px; }
                .button-group { flex-direction: column; align-items: center; }
                .btn-primary, .btn-secondary { min-width: 100%; }
            }
        </style>
    </head>
    <body>
        <div class="chilean-header">
            <div class="header-content">
                <h1>📝 Resumen de Texto</h1>
                <p>Inteligencia Artificial para Chile 🇨🇱</p>
            </div>
        </div>
        
        <div class="main-container">
            <div class="playground-card">
                <div class="input-section">
                    <h2>🚀 Crea Resúmenes Inteligentes</h2>
                    <p>Pega aquí el texto que quieres resumir y obtén un resumen profesional en segundos</p>
                    
                    <div class="textarea-container">
                        <textarea id="textInput" placeholder="📄 Pega aquí tu texto...&#10;&#10;Ejemplo:&#10;El marketing digital ha revolucionado la forma en que las empresas se conectan con sus clientes. En Chile, el 87% de la población usa internet regularmente, lo que representa una oportunidad enorme para las PyMEs..."></textarea>
                    </div>
                    
                    <div class="button-group">
                        <button onclick="summarizeText()" class="btn-primary">
                            🚀 Crear Resumen
                        </button>
                        <button onclick="clearAll()" class="btn-secondary">
                            🗑️ Limpiar Todo
                        </button>
                    </div>
                </div>
                
                <div class="output-section" id="outputSection">
                    <h3>📋 Resumen Generado</h3>
                    <div class="summary-content" id="summaryOutput"></div>
                    
                    <div class="stats-section">
                        <h3>📊 Estadísticas del Resumen</h3>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-number" id="originalWords">0</div>
                                <div class="stat-label">Palabras Originales</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-number" id="summaryWords">0</div>
                                <div class="stat-label">Palabras del Resumen</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-number" id="compressionRatio">0%</div>
                                <div class="stat-label">Ratio de Compresión</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chilean-footer">
            <p>🇨🇱 Desarrollado en Chile con ❤️ y Inteligencia Artificial</p>
        </div>
        
        <div class="back-link">
            <a href="/dashboard">← Volver al Dashboard</a>
        </div>
        
        <script>
            async function summarizeText() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) {
                    alert('🇨🇱 ¡Por favor ingresa texto para resumir!');
                    return;
                }
                
                const outputSection = document.getElementById('outputSection');
                const summaryOutput = document.getElementById('summaryOutput');
                
                // Mostrar sección de salida
                outputSection.style.display = 'block';
                
                // Mostrar loading
                summaryOutput.innerHTML = '<div class="loading">⏳ Generando resumen inteligente...</div>';
                
                // Calcular estadísticas originales
                const originalWords = text.split('\\s+').filter(word => word.length > 0).length;
                document.getElementById('originalWords').textContent = originalWords;
                
                try {
                    const response = await fetch('/openai/summarize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text_for_summarization: text })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        // Mostrar resumen
                        summaryOutput.textContent = data.summary;
                        
                        // Calcular estadísticas del resumen
                        const summaryWords = data.summary.split('\\s+').filter(word => word.length > 0).length;
                        const compressionRatio = Math.round(((originalWords - summaryWords) / originalWords) * 100);
                        
                        document.getElementById('summaryWords').textContent = summaryWords;
                        document.getElementById('compressionRatio').textContent = compressionRatio + '%';
                        
                        // Scroll suave a la sección de salida
                        outputSection.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        summaryOutput.innerHTML = '❌ <strong>Error:</strong> ' + data.detail;
                    }
                } catch (error) {
                    summaryOutput.innerHTML = '❌ <strong>Error de conexión:</strong> No se pudo conectar con el servidor';
                }
            }
            
            function clearAll() {
                document.getElementById('textInput').value = '';
                document.getElementById('outputSection').style.display = 'none';
                document.getElementById('originalWords').textContent = '0';
                document.getElementById('summaryWords').textContent = '0';
                document.getElementById('compressionRatio').textContent = '0%';
            }
            
            // Auto-resize textarea
            document.getElementById('textInput').addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 400) + 'px';
            });
            
            // Contador de caracteres en tiempo real
            document.getElementById('textInput').addEventListener('input', function() {
                const charCount = this.value.length;
                const wordCount = this.value.split('\\s+').filter(word => word.length > 0).length;
                
                // Actualizar placeholder con contador
                if (charCount > 0) {
                    this.placeholder = `📄 Texto ingresado: ${wordCount} palabras, ${charCount} caracteres`;
                }
            });
        </script>
    </body>
    </html>
    """)

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

# =============================
# EVALUACIÓN Y TESTING DE RAG
# =============================

def _load_eval_data():
    """Carga los datos de evaluación desde los archivos JSONL."""
    eval_data = {
        "answerable": [],
        "unanswerable": [],
        "reports": {}
    }
    
    try:
        # Usar ruta relativa simple
        eval_dir = Path("eval")
        print(f"Intentando cargar desde: {eval_dir.absolute()}")
        
        # Cargar preguntas answerable
        answerable_path = eval_dir / "answerable.jsonl"
        if answerable_path.exists():
            with open(answerable_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        eval_data["answerable"].append(json.loads(line))
            print(f"Cargadas {len(eval_data['answerable'])} preguntas answerable")
        else:
            print(f"Archivo no encontrado: {answerable_path.absolute()}")
        
        # Cargar preguntas unanswerable
        unanswerable_path = eval_dir / "unanswerable.jsonl"
        if unanswerable_path.exists():
            with open(unanswerable_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        eval_data["unanswerable"].append(json.loads(line))
            print(f"Cargadas {len(eval_data['unanswerable'])} preguntas unanswerable")
        else:
            print(f"Archivo no encontrado: {unanswerable_path.absolute()}")
        
        # Cargar reportes
        report_answerable_path = eval_dir / "report_answerable.json"
        if report_answerable_path.exists():
            with open(report_answerable_path, 'r', encoding='utf-8') as f:
                eval_data["reports"]["answerable"] = json.load(f)
            print(f"Reporte answerable cargado")
        else:
            print(f"Archivo no encontrado: {report_answerable_path.absolute()}")
        
        report_unanswerable_path = eval_dir / "report_unanswerable.json"
        if report_unanswerable_path.exists():
            with open(report_unanswerable_path, 'r', encoding='utf-8') as f:
                eval_data["reports"]["unanswerable"] = json.load(f)
            print(f"Reporte unanswerable cargado")
        else:
            print(f"Archivo no encontrado: {report_unanswerable_path.absolute()}")
                
        print(f"Datos de evaluación cargados: {len(eval_data['answerable'])} answerable, {len(eval_data['unanswerable'])} unanswerable")
                
    except Exception as e:
        print(f"Error cargando datos de evaluación: {e}")
        import traceback
        traceback.print_exc()
    
    return eval_data

@app.get("/eval/data")
async def get_eval_data():
    """Retorna todos los datos de evaluación."""
    print("Endpoint /eval/data llamado")
    data = _load_eval_data()
    print(f"Retornando datos: {len(data['answerable'])} answerable, {len(data['unanswerable'])} unanswerable")
    return data

@app.post("/eval/test")
async def test_eval_question(request: dict):
    """Testea una pregunta específica del conjunto de evaluación."""
    try:
        question = request.get("question", "")
        expected = request.get("expected", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="Pregunta requerida")
        
        # Obtener respuesta del RAG
        rag_response = rag_router.invoke({"question": question})
        
        # Calcular métricas básicas
        response_length = len(rag_response)
        expected_length = len(expected) if expected else 0
        
        return {
            "question": question,
            "expected": expected,
            "response": rag_response,
            "metrics": {
                "response_length": response_length,
                "expected_length": expected_length,
                "length_ratio": round(response_length / expected_length, 2) if expected_length > 0 else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en testing: {str(e)}")

@app.get("/eval/playground/", response_class=HTMLResponse)
async def eval_playground():
    """Interfaz para testing y evaluación de RAG."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧪 Testing RAG - Evaluación de Respuestas</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .header {
                background: linear-gradient(135deg, #d52b1e 0%, #f4a460 50%, #ffffff 100%);
                padding: 30px 0;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .header h1 {
                font-size: 3rem;
                color: #1e3c72;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .header p {
                font-size: 1.2rem;
                color: #2a5298;
                font-weight: 500;
            }
            
            .main-container {
                max-width: 1400px;
                margin: 40px auto;
                padding: 0 20px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .stat-card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }
            
            .stat-number {
                font-size: 3rem;
                font-weight: 800;
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .stat-label {
                color: #6c757d;
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            .testing-section {
                background: white;
                border-radius: 25px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
                margin-bottom: 30px;
            }
            
            .section-header {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 30px;
                border-bottom: 1px solid #dee2e6;
            }
            
            .section-header h2 {
                color: #1e3c72;
                font-size: 2rem;
                margin-bottom: 10px;
            }
            
            .section-header p {
                color: #6c757d;
                font-size: 1.1rem;
            }
            
            .content-section {
                padding: 30px;
            }
            
            .question-selector {
                margin-bottom: 30px;
            }
            
            .question-selector select {
                width: 100%;
                padding: 15px;
                border: 2px solid #e9ecef;
                border-radius: 15px;
                font-size: 16px;
                margin-bottom: 20px;
            }
            
            .question-display {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                border-left: 4px solid #667eea;
            }
            
            .question-display h4 {
                color: #1e3c72;
                margin-bottom: 10px;
                font-size: 1.2rem;
            }
            
            .expected-answer {
                background: #e8f5e8;
                padding: 15px;
                border-radius: 10px;
                margin-top: 10px;
                border-left: 4px solid #28a745;
            }
            
            .expected-answer h5 {
                color: #155724;
                margin-bottom: 8px;
                font-size: 1rem;
            }
            
            .test-button {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }
            
            .test-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
            }
            
            .unanswerable-test {
                background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
                box-shadow: 0 8px 25px rgba(220, 53, 69, 0.3) !important;
            }
            
            .unanswerable-test:hover {
                box-shadow: 0 12px 35px rgba(220, 53, 69, 0.4) !important;
            }
            
            .question-type-indicator {
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .indicator-answerable {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                color: #155724;
                border-left: 5px solid #28a745;
            }
            
            .indicator-unanswerable {
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                color: #721c24;
                border-left: 5px solid #dc3545;
            }
            
            .results-section {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                border-left: 4px solid #17a2b8;
                display: none;
            }
            
            .results-section h4 {
                color: #1e3c72;
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            
            .rag-response {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }
            
            .metric-item {
                background: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid #dee2e6;
            }
            
            .metric-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .metric-label {
                color: #6c757d;
                font-size: 0.9rem;
            }
            
            .back-link {
                text-align: center;
                margin-top: 30px;
            }
            
            .back-link a {
                display: inline-block;
                background: linear-gradient(135deg, #d52b1e 0%, #f4a460 100%);
                color: white;
                text-decoration: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(213, 43, 30, 0.3);
                margin: 10px;
            }
            
            .back-link a:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(213, 43, 30, 0.4);
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #667eea;
            }
            
            .loading::after {
                content: '';
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #667eea;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 1s linear infinite;
                margin-left: 10px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .header h1 { font-size: 2rem; }
                .stats-grid { grid-template-columns: 1fr; }
                .content-section { padding: 20px; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Testing RAG</h1>
            <p>Evaluación y Testing de Respuestas - Sistema de Evaluación LangChain</p>
        </div>
        
        <div class="main-container">
            <!-- Estadísticas Generales -->
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-number">...</div>
                    <div class="stat-label">Preguntas Answerable</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">...</div>
                    <div class="stat-label">Preguntas Unanswerable</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">...</div>
                    <div class="stat-label">Tasa de Respuesta</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">...</div>
                    <div class="stat-label">Latencia Promedio</div>
                </div>
            </div>
            
            <!-- Sección de Testing -->
            <div class="testing-section">
                <div class="section-header">
                    <h2>🔬 Testing de Preguntas</h2>
                    <p>Selecciona una pregunta del conjunto de evaluación y compara la respuesta del RAG con la esperada</p>
                </div>
                
                <div class="content-section">
                    <div class="question-selector">
                        <select id="questionSelect" onchange="loadQuestion()">
                            <option value="">-- Selecciona una pregunta para testear --</option>
                        </select>
                    </div>
                    
                    <div class="question-display" id="questionDisplay" style="display: none;">
                        <div class="question-type-indicator" id="questionTypeIndicator">
                            <!-- Se llenará dinámicamente -->
                        </div>
                        
                        <h4>📝 Pregunta Seleccionada:</h4>
                        <p id="selectedQuestion"></p>
                        
                        <div class="expected-answer">
                            <h5>✅ Respuesta Esperada:</h5>
                            <p id="expectedAnswer"></p>
                        </div>
                        
                        <button onclick="testQuestion()" class="test-button" id="testButton">
                            🚀 Testear Respuesta RAG
                        </button>
                    </div>
                    
                    <div class="results-section" id="resultsSection">
                        <h4>📊 Resultados del Test</h4>
                        <div class="rag-response">
                            <h5>🤖 Respuesta del RAG:</h5>
                            <p id="ragResponse"></p>
                        </div>
                        
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <div class="metric-value" id="responseLength">0</div>
                                <div class="metric-label">Longitud Respuesta</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value" id="expectedLength">0</div>
                                <div class="metric-label">Longitud Esperada</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value" id="lengthRatio">0</div>
                                <div class="metric-label">Ratio Longitud</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="back-link">
            <a href="/dashboard">← Volver al Dashboard</a>
        </div>
        
        <script>
            let evalData = {};
            let currentQuestion = null;
            
            // Cargar datos de evaluación al iniciar
            async function loadEvalData() {
                try {
                    const response = await fetch('/eval/data');
                    evalData = await response.json();
                    updateStats();
                    populateQuestionSelector();
                } catch (error) {
                    console.error('Error cargando datos de evaluación:', error);
                }
            }
            
            // Actualizar estadísticas
            function updateStats() {
                const stats = evalData.reports || {};
                const answerable = stats.answerable || {};
                const unanswerable = stats.unanswerable || {};
                
                const statsGrid = document.getElementById('statsGrid');
                const statCards = statsGrid.querySelectorAll('.stat-card');
                
                if (statCards.length >= 4) {
                    statCards[0].querySelector('.stat-number').textContent = evalData.answerable?.length || 0;
                    statCards[1].querySelector('.stat-number').textContent = evalData.unanswerable?.length || 0;
                    statCards[2].querySelector('.stat-number').textContent = (answerable.answerable_rate * 100).toFixed(1) + '%';
                    statCards[3].querySelector('.stat-number').textContent = (answerable.avg_latency_s || 0).toFixed(2) + 's';
                }
            }
            
            // Poblar selector de preguntas
            function populateQuestionSelector() {
                const select = document.getElementById('questionSelect');
                const answerableQuestions = evalData.answerable || [];
                const unanswerableQuestions = evalData.unanswerable || [];
                
                // Limpiar opciones existentes
                select.innerHTML = '<option value="">-- Selecciona una pregunta para testear --</option>';
                
                // Agregar preguntas answerable (Respondibles)
                if (answerableQuestions.length > 0) {
                    const answerableGroup = document.createElement('optgroup');
                    answerableGroup.label = '🧪 Preguntas Respondibles (' + answerableQuestions.length + ')';
                    
                    answerableQuestions.forEach((item, index) => {
                        const option = document.createElement('option');
                        option.value = 'answerable_' + index;
                        option.textContent = '✅ ' + item.question.substring(0, 75) + '...';
                        answerableGroup.appendChild(option);
                    });
                    
                    select.appendChild(answerableGroup);
                }
                
                // Agregar preguntas unanswerable (Irrespondibles)
                if (unanswerableQuestions.length > 0) {
                    const unanswerableGroup = document.createElement('optgroup');
                    unanswerableGroup.label = '❓ Preguntas Irrespondibles (' + unanswerableQuestions.length + ')';
                    
                    unanswerableQuestions.forEach((item, index) => {
                        const option = document.createElement('option');
                        option.value = 'unanswerable_' + index;
                        option.textContent = '❌ ' + item.question.substring(0, 75) + '...';
                        unanswerableGroup.appendChild(option);
                    });
                    
                    select.appendChild(unanswerableGroup);
                }
            }
            
            // Cargar pregunta seleccionada
            function loadQuestion() {
                const select = document.getElementById('questionSelect');
                const questionDisplay = document.getElementById('questionDisplay');
                
                if (select.value === '') {
                    questionDisplay.style.display = 'none';
                    return;
                }
                
                const [type, indexStr] = select.value.split('_');
                const index = parseInt(indexStr);
                
                if (type === 'answerable') {
                    currentQuestion = evalData.answerable[index];
                    currentQuestion.type = 'answerable';
                    // Cambiar el estilo del botón para preguntas respondibles
                    document.getElementById('testButton').textContent = '🚀 Testear Respuesta RAG';
                    document.getElementById('testButton').className = 'test-button';
                    
                    // Mostrar indicador de pregunta respondible
                    document.getElementById('questionTypeIndicator').innerHTML = 
                        '🧪 <strong>Pregunta Respondible</strong> - El RAG debe proporcionar una respuesta basada en los documentos';
                    document.getElementById('questionTypeIndicator').className = 'question-type-indicator indicator-answerable';
                    
                } else if (type === 'unanswerable') {
                    currentQuestion = evalData.unanswerable[index];
                    currentQuestion.type = 'unanswerable';
                    // Cambiar el estilo del botón para preguntas irrespondibles
                    document.getElementById('testButton').textContent = '🚫 Validar Abstinencia RAG';
                    document.getElementById('testButton').className = 'test-button unanswerable-test';
                    
                    // Mostrar indicador de pregunta irrespondible
                    document.getElementById('questionTypeIndicator').innerHTML = 
                        '❓ <strong>Pregunta Irrespondible</strong> - El RAG debe abstenerse de responder';
                    document.getElementById('questionTypeIndicator').className = 'question-type-indicator indicator-unanswerable';
                }
                
                document.getElementById('selectedQuestion').textContent = currentQuestion.question;
                document.getElementById('expectedAnswer').textContent = currentQuestion.expected;
                questionDisplay.style.display = 'block';
                
                // Ocultar resultados anteriores
                document.getElementById('resultsSection').style.display = 'none';
            }
            
            // Testear pregunta
            async function testQuestion() {
                if (!currentQuestion) return;
                
                const resultsSection = document.getElementById('resultsSection');
                const ragResponse = document.getElementById('ragResponse');
                const testButton = document.getElementById('testButton');
                
                // Deshabilitar botón durante el testing
                testButton.disabled = true;
                testButton.textContent = '⏳ Procesando...';
                
                resultsSection.style.display = 'block';
                ragResponse.innerHTML = '<div class="loading">⏳ Generando respuesta RAG...</div>';
                
                try {
                    const response = await fetch('/eval/test', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            question: currentQuestion.question,
                            expected: currentQuestion.expected
                        })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        // Mostrar respuesta RAG
                        ragResponse.innerHTML = data.response;
                        
                        // Actualizar métricas
                        document.getElementById('responseLength').textContent = data.metrics.response_length;
                        document.getElementById('expectedLength').textContent = data.metrics.expected_length;
                        document.getElementById('lengthRatio').textContent = data.metrics.length_ratio;
                        
                        // Scroll suave a resultados
                        resultsSection.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        ragResponse.innerHTML = '❌ <strong>Error:</strong> ' + data.detail;
                    }
                } catch (error) {
                    ragResponse.innerHTML = '❌ <strong>Error de conexión:</strong> No se pudo conectar con el servidor';
                } finally {
                    // Restaurar botón
                    testButton.disabled = false;
                    if (currentQuestion.type === 'answerable') {
                        testButton.textContent = '🚀 Testear Respuesta RAG';
                    } else {
                        testButton.textContent = '🚫 Validar Abstinencia RAG';
                    }
                }
            }
            
            // Cargar datos al iniciar
            loadEvalData();
        </script>
    </body>
    </html>
    """)

# Simple RAG endpoints
@app.post("/rag/query")
async def rag_query(question: RAGInput):
    """Query the RAG system with a question."""
    try:
        response = rag_router.invoke(question.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/playground/")
async def rag_playground():
    """Modern and beautiful playground interface for RAG queries."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔍 RAG System Playground</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 40px;
                padding: 30px 0;
            }
            
            .header h1 {
                font-size: 3.5rem;
                margin-bottom: 15px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                background: linear-gradient(45deg, #fff, #f0f8ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .header p {
                font-size: 1.3rem;
                opacity: 0.9;
                max-width: 600px;
                margin: 0 auto;
                line-height: 1.6;
            }
            
            .main-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 25px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                margin-bottom: 30px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .input-section {
                margin-bottom: 30px;
            }
            
            .input-group {
                position: relative;
                margin-bottom: 20px;
            }
            
            .input-group label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #555;
                font-size: 1.1rem;
            }
            
            .question-input {
                width: 100%;
                padding: 20px 25px;
                font-size: 1.1rem;
                border: 2px solid #e1e5e9;
                border-radius: 15px;
                transition: all 0.3s ease;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .question-input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }
            
            .question-input::placeholder {
                color: #aaa;
                font-style: italic;
            }
            
            .button-group {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 15px 30px;
                font-size: 1.1rem;
                font-weight: 600;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 10px;
                text-decoration: none;
                min-width: 160px;
                justify-content: center;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn-primary:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
            }
            
            .btn-secondary:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(108, 117, 125, 0.6);
            }
            
            .btn:active {
                transform: translateY(-1px);
            }
            
            .result-section {
                margin-top: 30px;
                display: none;
            }
            
            .result-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 20px;
                padding: 30px;
                border-left: 5px solid #667eea;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
                animation: slideIn 0.5s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-header {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #e9ecef;
            }
            
            .result-icon {
                font-size: 2rem;
                color: #667eea;
            }
            
            .result-title {
                font-size: 1.5rem;
                font-weight: 700;
                color: #333;
                margin: 0;
            }
            
            .result-content {
                font-size: 1.1rem;
                line-height: 1.8;
                color: #555;
                background: white;
                padding: 20px;
                border-radius: 15px;
                border: 1px solid #e9ecef;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
                color: #667eea;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error-card {
                background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
                border-left-color: #e53e3e;
            }
            
            .error-icon {
                color: #e53e3e;
            }
            
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
            }
            
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 15px;
                display: block;
            }
            
            .feature-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .feature-desc {
                font-size: 0.95rem;
                opacity: 0.9;
                line-height: 1.5;
            }
            
            .back-link {
                text-align: center;
                margin-top: 30px;
            }
            
            .back-link a {
                color: white;
                text-decoration: none;
                font-weight: 600;
                font-size: 1.1rem;
                padding: 12px 25px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 25px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
                display: inline-block;
            }
            
            .back-link a:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                }
                
                .header h1 {
                    font-size: 2.5rem;
                }
                
                .main-card {
                    padding: 25px;
                }
                
                .button-group {
                    flex-direction: column;
                    align-items: center;
                }
                
                .btn {
                    width: 100%;
                    max-width: 300px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 RAG System Playground</h1>
                <p>Haz preguntas inteligentes sobre tus documentos y obtén respuestas basadas en tu base de conocimientos</p>
            </div>
            
            <div class="main-card">
                <div class="input-section">
                    <div class="input-group">
                        <label for="questionInput">
                            <i class="fas fa-question-circle"></i> ¿Qué te gustaría saber?
                        </label>
                        <input 
                            type="text" 
                            id="questionInput" 
                            class="question-input" 
                            placeholder="Ej: ¿Cuáles son las tendencias principales del marketing digital en 2024?"
                            onkeypress="if(event.key==='Enter') askQuestion()"
                        >
                    </div>
                    
                    <div class="button-group">
                        <button onclick="askQuestion()" class="btn btn-primary">
                            <i class="fas fa-search"></i> Buscar Respuesta
                        </button>
                        <button onclick="clearQuestion()" class="btn btn-secondary">
                            <i class="fas fa-eraser"></i> Limpiar
                        </button>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Buscando en tu base de conocimientos...</p>
                </div>
                
                <div class="result-section" id="resultSection">
                    <div class="result-card" id="resultCard">
                        <div class="result-header">
                            <i class="fas fa-lightbulb result-icon" id="resultIcon"></i>
                            <h3 class="result-title" id="resultTitle">Respuesta</h3>
                        </div>
                        <div class="result-content" id="resultContent">
                            <!-- El contenido se insertará aquí -->
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="features-grid">
                <div class="feature-card">
                    <i class="fas fa-brain feature-icon"></i>
                    <div class="feature-title">IA Inteligente</div>
                    <div class="feature-desc">Respuestas basadas en GPT-4o con contexto de tus documentos</div>
                </div>
                <div class="feature-card">
                    <i class="fas fa-database feature-icon"></i>
                    <div class="feature-title">Base de Conocimientos</div>
                    <div class="feature-desc">Acceso a toda tu información procesada y vectorizada</div>
                </div>
                <div class="feature-card">
                    <i class="fas fa-bolt feature-icon"></i>
                    <div class="feature-title">Respuesta Rápida</div>
                    <div class="feature-desc">Resultados instantáneos con búsqueda semántica avanzada</div>
                </div>
            </div>
            
            <div class="back-link">
                <a href="/dashboard">
                    <i class="fas fa-arrow-left"></i> Volver al Dashboard
                </a>
            </div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('questionInput').value.trim();
                if (!question) {
                    showError('Por favor, ingresa una pregunta');
                    return;
                }
                
                // Mostrar loading
                showLoading(true);
                hideResult();
                
                try {
                    const response = await fetch('/rag/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: question})
                    });
                    
                    const data = await response.json();
                    showLoading(false);
                    
                    if (response.ok) {
                        showSuccess(data.response);
                    } else {
                        showError(data.detail || 'Error en la consulta');
                    }
                } catch (error) {
                    showLoading(false);
                    showError('Error de conexión. Verifica tu conexión a internet.');
                }
            }
            
            function showLoading(show) {
                const loading = document.getElementById('loading');
                loading.style.display = show ? 'block' : 'none';
            }
            
            function showSuccess(response) {
                const resultSection = document.getElementById('resultSection');
                const resultCard = document.getElementById('resultCard');
                const resultIcon = document.getElementById('resultIcon');
                const resultTitle = document.getElementById('resultTitle');
                const resultContent = document.getElementById('resultContent');
                
                resultIcon.className = 'fas fa-check-circle result-icon';
                resultIcon.style.color = '#28a745';
                resultTitle.textContent = 'Respuesta Encontrada';
                resultContent.innerHTML = formatResponse(response);
                
                resultCard.className = 'result-card';
                resultSection.style.display = 'block';
                
                // Scroll suave al resultado
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            function showError(message) {
                const resultSection = document.getElementById('resultSection');
                const resultCard = document.getElementById('resultCard');
                const resultIcon = document.getElementById('resultIcon');
                const resultTitle = document.getElementById('resultTitle');
                const resultContent = document.getElementById('resultContent');
                
                resultIcon.className = 'fas fa-exclamation-triangle result-icon error-icon';
                resultTitle.textContent = 'Error';
                resultContent.innerHTML = '<p style="color: #e53e3e; font-weight: 600;">' + message + '</p>';
                
                resultCard.className = 'result-card error-card';
                resultSection.style.display = 'block';
                
                resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            function hideResult() {
                document.getElementById('resultSection').style.display = 'none';
            }
            
            function clearQuestion() {
                document.getElementById('questionInput').value = '';
                hideResult();
                document.getElementById('questionInput').focus();
            }
            
            function formatResponse(response) {
                // Formatear la respuesta para mejor legibilidad
                if (typeof response === 'string') {
                    // Dividir por líneas y crear párrafos
                    const paragraphs = response.split('\\n\\n').filter(p => p.trim());
                    return paragraphs.map(p => '<p>' + p.trim() + '</p>').join('');
                }
                return '<p>' + response + '</p>';
            }
            
            // Auto-focus en el input al cargar la página
            window.onload = function() {
                document.getElementById('questionInput').focus();
            };
        </script>
    </body>
    </html>
    """)

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
async def get_ingest_status_json():
    """Retorna el estado actual de la colección Qdrant como JSON (para frontend)."""
    try:
        stats = _compute_corpus_stats()
        if "error" in stats:
            return JSONResponse({"status": "error", "detail": stats["error"], "collection_stats": {"total_files": 0, "total_chunks": 0, "by_type": {}, "samples": []}}, status_code=200)
        return JSONResponse({"status": "ok", "collection_stats": stats})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

@app.get("/ingest/status/html", response_class=HTMLResponse)
async def get_ingest_status_html():
    """Retorna el estado actual de la colección Qdrant con formato HTML bonito."""
    try:
        stats = _compute_corpus_stats()
        if "error" in stats:
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error en Estadísticas</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
                    .error-container {{ max-width: 600px; margin: 50px auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 30px; text-align: center; }}
                    .error-icon {{ font-size: 64px; margin-bottom: 20px; }}
                    .error-title {{ color: #dc3545; font-size: 24px; margin-bottom: 15px; }}
                    .error-message {{ color: #666; margin-bottom: 25px; }}
                    .back-link a {{ color: white; text-decoration: none; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1 class="error-title">Error al cargar estadísticas</h1>
                    <p class="error-message">{stats["error"]}</p>
                    <a href="/dashboard" class="btn">← Volver al Dashboard</a>
                </div>
                <div class="back-link">
                    <a href="/dashboard">← Volver al Dashboard</a>
                </div>
            </body>
            </html>
            """)

        # Crear HTML bonito para las estadísticas
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
                .stat-card {{ background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 5px solid #667eea; transition: transform 0.3s ease; }}
                .stat-card:hover {{ transform: translateY(-5px); }}
                .stat-number {{ font-size: 48px; font-weight: bold; color: #667eea; margin: 10px 0; }}
                .stat-label {{ font-size: 18px; color: #666; margin-bottom: 15px; }}
                .stat-description {{ color: #888; font-size: 14px; }}
                .files-section {{ padding: 30px; background: #f8f9fa; }}
                .files-section h3 {{ color: #333; margin-bottom: 20px; font-size: 24px; }}
                .file-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .file-item {{ background: white; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; transition: transform 0.2s ease; }}
                .file-item:hover {{ transform: scale(1.02); }}
                .file-name {{ font-weight: bold; color: #333; margin-bottom: 8px; }}
                .file-type {{ color: #666; font-size: 14px; }}
                .file-icon {{ font-size: 24px; margin-right: 10px; }}
                .back-link {{ text-align: center; margin-top: 20px; }}
                .back-link a {{ color: white; text-decoration: none; font-weight: bold; font-size: 16px; }}
                .refresh-btn {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; border: none; padding: 12px 24px; border-radius: 25px; cursor: pointer; font-size: 16px; margin: 10px; }}
                .refresh-btn:hover {{ opacity: 0.9; }}
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
                        <div class="stat-label">📁 Total de Archivos</div>
                        <div class="stat-number">{stats.get('total_files', 0)}</div>
                        <div class="stat-description">Documentos procesados</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🧩 Total de Chunks</div>
                        <div class="stat-number">{stats.get('total_chunks', 0)}</div>
                        <div class="stat-description">Fragmentos de texto</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🎯 Tipos de Archivo</div>
                        <div class="stat-number">{len(stats.get('by_type', {}))}</div>
                        <div class="stat-description">Formatos soportados</div>
                    </div>
                </div>
                
                <div class="files-section">
                    <h3>📋 Desglose por Tipo de Archivo</h3>
                    <div class="stats-grid">
        """
        
        # Agregar estadísticas por tipo
        for file_type, count in stats.get('by_type', {}).items():
            if file_type == 'pdf':
                icon = "📄"
                description = "Documentos PDF"
            elif file_type in ['doc', 'docx']:
                icon = "📝"
                description = "Documentos Word"
            elif file_type == 'txt':
                icon = "📄"
                description = "Archivos de texto"
            else:
                icon = "📁"
                description = f"Archivos {file_type.upper()}"
            
            stats_html += f"""
                        <div class="stat-card">
                            <div class="stat-label">{icon} {file_type.upper()}</div>
                            <div class="stat-number">{count}</div>
                            <div class="stat-description">{description}</div>
                        </div>
            """
        
        stats_html += """
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
                icon = "📁"
                
            stats_html += f"""
                        <div class="file-item">
                            <div class="file-name">{icon} {file_name}</div>
                            <div class="file-type">Tipo: {file_type.upper()}</div>
                        </div>
            """
        
        stats_html += """
                    </div>
                </div>
                
                <div style="text-align: center; padding: 20px;">
                    <button onclick="location.reload()" class="refresh-btn">🔄 Actualizar Estadísticas</button>
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
            .chat-header p { margin: 10px 0 0 0; font-size: 18px; opacity: 0.9; }
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
# Endpoint de prueba simple
@app.get("/test")
async def test_endpoint():
    """Endpoint de prueba simple."""
    return {"message": "Servidor funcionando correctamente", "status": "ok"}

@app.get("/stats", response_class=HTMLResponse)
async def stats_page():
    """Página bonita para mostrar estadísticas del sistema."""
    try:
        # Obtener estadísticas
        try:
            stats = _compute_corpus_stats()
            if "error" in stats:
                stats = {"total_files": 0, "total_chunks": 0, "by_type": {}, "samples": []}
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            stats = {"total_files": 0, "total_chunks": 0, "by_type": {}, "samples": [], "error": str(e)}
        
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
                
                {f'''
                <div class="files-section" style="background: #fff3cd; border-left: 5px solid #ffc107;">
                    <h3>⚠️ Problema de Conexión</h3>
                    <p>No se pudieron cargar las estadísticas completas. Error: {stats.get('error', 'Desconocido')}</p>
                    <p><strong>Verifica que:</strong></p>
                    <ul>
                        <li>El archivo .env esté configurado correctamente</li>
                        <li>Las credenciales de Qdrant sean válidas</li>
                        <li>La conexión a internet esté funcionando</li>
                    </ul>
                </div>
                ''' if stats.get('error') else ''}
                
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
                icon = "📁"
                
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