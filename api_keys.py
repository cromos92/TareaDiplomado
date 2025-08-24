"""
Carga y expone de forma global las API keys para proveedores usados en el
Diplomado RAG: OpenAI (ChatGPT), Qdrant y LangChain.

Uso rápido en notebooks:

    from api_keys import API_KEYS, load_api_keys, get_api_key

    # (opcional) recargar desde .env si cambiaste variables
    load_api_keys(override_env=False)

    # acceder a valores
    openai_key = API_KEYS["openai"]["api_key"]
    qdrant_url = API_KEYS["qdrant"]["url"]
    langchain_key = get_api_key("langchain")

Las variables se leen de variables de entorno y del archivo .env si existe.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv


def _read_env() -> None:
    """Carga variables desde el primer .env encontrado sin sobreescribir el entorno."""
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path or None, override=False)


def load_api_keys(override_env: bool = False) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Carga (y opcionalmente sobreescribe) variables de entorno desde .env y
    devuelve un diccionario con claves agrupadas por proveedor.

    - override_env=False: respeta variables ya exportadas en el entorno.
    - override_env=True: valores del .env sobre-escriben el entorno actual.
    """
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path or None, override=override_env)

    keys: Dict[str, Dict[str, Optional[str]]] = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),  # útil para Azure u OSS gateways
            "organization": os.getenv("OPENAI_ORG"),
            "project": os.getenv("OPENAI_PROJECT"),
        },
        "qdrant": {
            "api_key": os.getenv("QDRANT_API_KEY"),
            # Prioriza URL completa; alternativamente host/port
            "url": os.getenv("QDRANT_URL"),
            "host": os.getenv("QDRANT_HOST"),
            "port": os.getenv("QDRANT_PORT"),
            "collection": os.getenv("QDRANT_COLLECTION"),
        },
        "langchain": {
            # Clave para LangSmith (telemetría/observabilidad de LangChain)
            "api_key": os.getenv("LANGCHAIN_API_KEY"),
            "endpoint": os.getenv("LANGCHAIN_ENDPOINT"),
            "project": os.getenv("LANGCHAIN_PROJECT"),
            # Variables de tracing opcionales
            "tracing_v2": os.getenv("LANGCHAIN_TRACING_V2"),
            "timeout": os.getenv("LANGCHAIN_TIMEOUT"),
        },
    }

    # Actualiza variable global al llamar explícitamente load_api_keys.
    global API_KEYS
    API_KEYS = keys
    return keys


def get_api_key(provider: str, key_name: str = "api_key") -> Optional[str]:
    """Devuelve un valor de clave específico, p.ej. get_api_key("openai")."""
    provider = provider.lower()
    data = API_KEYS.get(provider, {})  # type: ignore[arg-type]
    return data.get(key_name) if isinstance(data, dict) else None


# Carga inicial al importar el módulo
_read_env()
API_KEYS: Dict[str, Dict[str, Optional[str]]] = load_api_keys(override_env=False)


