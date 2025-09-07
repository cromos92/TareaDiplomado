"""
Configuración de logging con structlog para FastAPI
"""

import sys
import structlog
from structlog.stdlib import LoggerFactory
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, StackInfoRenderer


def configure_logging():
    """
    Configura structlog para logging estructurado
    """
    
    # Configurar procesadores
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        TimeStamper(fmt="ISO"),
        StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Usar JSON en producción, console en desarrollo
    if sys.stdout.isatty():
        # Desarrollo - output legible para humanos
        processors.append(ConsoleRenderer(colors=True))
    else:
        # Producción - JSON estructurado
        processors.append(JSONRenderer())
    
    # Configurar structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """
    Obtiene un logger configurado
    """
    return structlog.get_logger(name)
