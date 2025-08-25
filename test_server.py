#!/usr/bin/env python3
"""
Script de prueba para verificar que el servidor funcione correctamente.
"""

import uvicorn
from app.server import app

if __name__ == "__main__":
    print("🚀 Iniciando servidor de prueba...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")
        import traceback
        traceback.print_exc()
