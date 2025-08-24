#!/usr/bin/env python3
"""
Script de prueba para el endpoint de ingesta de documentos.
Este script demuestra cómo usar la nueva funcionalidad de ingesta web.
"""

import requests
import json
import os
from pathlib import Path

def test_ingest_endpoints():
    """Prueba los endpoints de ingesta."""
    base_url = "http://localhost:8000"
    
    print("🧪 Probando endpoints de ingesta...")
    
    # 1. Probar endpoint raíz
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint raíz funciona")
            if "ingest" in data.get("endpoints", {}):
                print("✅ Endpoints de ingesta están disponibles")
        else:
            print(f"❌ Endpoint raíz falló: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando al servidor: {e}")
        return
    
    # 2. Probar endpoint de estadísticas
    try:
        response = requests.get(f"{base_url}/ingest/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint de estadísticas funciona")
            if "collection_stats" in data:
                stats = data["collection_stats"]
                print(f"   📊 Archivos: {stats.get('total_files', 0)}")
                print(f"   📊 Chunks: {stats.get('total_chunks', 0)}")
        else:
            print(f"❌ Endpoint de estadísticas falló: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en estadísticas: {e}")
    
    # 3. Probar interfaz web
    try:
        response = requests.get(f"{base_url}/ingest/ui")
        if response.status_code == 200:
            print("✅ Interfaz web de ingesta funciona")
            if "Ingesta de Documentos" in response.text:
                print("   📱 HTML se carga correctamente")
        else:
            print(f"❌ Interfaz web falló: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en interfaz web: {e}")
    
    print("\n🎯 Para usar la funcionalidad:")
    print(f"   1. Interfaz web: {base_url}/ingest/ui")
    print(f"   2. API de subida: POST {base_url}/ingest/upload")
    print(f"   3. Estadísticas: GET {base_url}/ingest/status")
    print(f"   4. RAG: {base_url}/rag/playground/")

def test_file_upload_example():
    """Ejemplo de cómo usar la API de subida."""
    print("\n📤 Ejemplo de uso de la API de subida:")
    print("""
# Subir un PDF con chunking semántico
curl -X POST http://localhost:8000/ingest/upload \\
  -F "file=@documento.pdf" \\
  -F "chunker_type=semantic" \\
  -F "chunk_size=1200" \\
  -F "chunk_overlap=200"

# Subir un Word con chunking semántico
curl -X POST http://localhost:8000/ingest/upload \\
  -F "file=@documento.docx" \\
  -F "chunker_type=semantic" \\
  -F "chunk_size=1000" \\
  -F "chunk_overlap=150"

# Subir un TXT con chunking recursivo
curl -X POST http://localhost:8000/ingest/upload \\
  -F "file=@documento.txt" \\
  -F "chunker_type=recursive" \\
  -F "chunk_size=800" \\
  -F "chunk_overlap=120"
    """)

if __name__ == "__main__":
    print("🚀 Sistema de Ingesta Web - Pruebas")
    print("=" * 50)
    
    test_ingest_endpoints()
    test_file_upload_example()
    
    print("\n✨ ¡Implementación completada!")
    print("   Ahora puedes subir documentos directamente desde la web")
    print("   en lugar de usar comandos de línea.")
