"""
Módulo de datos para Farmacia IA
Incluye clientes para APIs externas y procesamiento de datos
"""

from .minsal_client import MinsalClient
from .normalizers import CommuneNormalizer

__version__ = "1.0.0"
__all__ = [
    "MinsalClient",
    "CommuneNormalizer",
]
