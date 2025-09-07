"""
Configuración de Sentry para monitoreo de errores
"""

import os
import logging
from typing import Dict, Any, Optional

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from django.conf import settings

logger = logging.getLogger(__name__)


def before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filtro de eventos antes de enviar a Sentry
    Permite filtrar o modificar eventos antes del envío
    """
    
    # Filtrar errores de desarrollo/testing
    if getattr(settings, 'DEBUG', False):
        return None
    
    # Filtrar errores conocidos que no son críticos
    if event.get('exception'):
        exc_info = hint.get('exc_info')
        if exc_info:
            exc_type, exc_value, exc_traceback = exc_info
            
            # Filtrar errores de conexión Redis en desarrollo
            if 'ConnectionError' in str(exc_type) and 'redis' in str(exc_value).lower():
                return None
            
            # Filtrar errores de rate limiting (son esperados)
            if 'RateLimitExceeded' in str(exc_type):
                return None
    
    # Filtrar requests de bots/crawlers
    request = event.get('request', {})
    user_agent = request.get('headers', {}).get('User-Agent', '')
    
    bot_patterns = ['bot', 'crawler', 'spider', 'scraper']
    if any(pattern in user_agent.lower() for pattern in bot_patterns):
        return None
    
    # Añadir contexto adicional
    event.setdefault('tags', {})
    event['tags']['environment'] = getattr(settings, 'ENVIRONMENT', 'unknown')
    event['tags']['service'] = 'farmacia-ia'
    
    # Añadir información del usuario si está disponible
    user = event.get('user', {})
    if not user.get('id') and hasattr(settings, 'SENTRY_USER_CONTEXT'):
        event['user'] = settings.SENTRY_USER_CONTEXT
    
    return event


def before_send_transaction(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filtro para transacciones de performance
    """
    
    # No enviar transacciones en desarrollo
    if getattr(settings, 'DEBUG', False):
        return None
    
    # Filtrar transacciones de health checks
    transaction_name = event.get('transaction', '')
    if any(pattern in transaction_name.lower() for pattern in ['health', 'metrics', 'ping']):
        return None
    
    # Solo enviar una muestra de transacciones para reducir volumen
    import random
    if random.random() > 0.1:  # 10% de sampling
        return None
    
    return event


def configure_sentry():
    """
    Configura Sentry para monitoreo de errores y performance
    """
    
    sentry_dsn = os.getenv('SENTRY_DSN')
    
    if not sentry_dsn:
        logger.info("SENTRY_DSN no configurado, Sentry deshabilitado")
        return
    
    # Configuración de logging para Sentry
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capturar logs de nivel INFO y superior
        event_level=logging.ERROR  # Solo enviar eventos para ERROR y superior
    )
    
    # Configurar Sentry
    sentry_sdk.init(
        dsn=sentry_dsn,
        
        # Integraciones
        integrations=[
            DjangoIntegration(
                transaction_style='url',
                middleware_spans=True,
                signals_spans=True,
                cache_spans=True,
            ),
            RedisIntegration(),
            SqlalchemyIntegration(),
            HttpxIntegration(),
            AsyncioIntegration(),
            sentry_logging,
        ],
        
        # Configuración de sampling
        traces_sample_rate=float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
        profiles_sample_rate=float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', '0.1')),
        
        # Filtros
        before_send=before_send,
        before_send_transaction=before_send_transaction,
        
        # Configuración de release y environment
        release=os.getenv('SENTRY_RELEASE', getattr(settings, 'VERSION', 'unknown')),
        environment=os.getenv('SENTRY_ENVIRONMENT', getattr(settings, 'ENVIRONMENT', 'development')),
        
        # Configuración de datos sensibles
        send_default_pii=False,  # No enviar PII por defecto
        
        # Configuración de performance
        enable_tracing=True,
        
        # Configuración adicional
        attach_stacktrace=True,
        max_breadcrumbs=50,
        
        # Tags globales
        _experiments={
            "profiles_sample_rate": float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', '0.1')),
        }
    )
    
    # Configurar contexto global
    sentry_sdk.set_tag("service", "farmacia-ia")
    sentry_sdk.set_tag("component", "django")
    
    logger.info(f"Sentry configurado correctamente para environment: {os.getenv('SENTRY_ENVIRONMENT', 'development')}")


def capture_message(message: str, level: str = 'info', **kwargs):
    """
    Captura un mensaje en Sentry con contexto adicional
    
    Args:
        message: Mensaje a capturar
        level: Nivel del mensaje (debug, info, warning, error, fatal)
        **kwargs: Contexto adicional
    """
    
    with sentry_sdk.push_scope() as scope:
        # Añadir contexto adicional
        for key, value in kwargs.items():
            scope.set_extra(key, value)
        
        sentry_sdk.capture_message(message, level)


def capture_exception(exception: Exception, **kwargs):
    """
    Captura una excepción en Sentry con contexto adicional
    
    Args:
        exception: Excepción a capturar
        **kwargs: Contexto adicional
    """
    
    with sentry_sdk.push_scope() as scope:
        # Añadir contexto adicional
        for key, value in kwargs.items():
            scope.set_extra(key, value)
        
        sentry_sdk.capture_exception(exception)


def set_user_context(user_id: str, email: str = None, **kwargs):
    """
    Establece contexto de usuario para Sentry
    
    Args:
        user_id: ID del usuario
        email: Email del usuario (opcional)
        **kwargs: Datos adicionales del usuario
    """
    
    user_data = {
        'id': user_id,
        **kwargs
    }
    
    if email:
        user_data['email'] = email
    
    sentry_sdk.set_user(user_data)


def set_conversation_context(conversation_id: str, user_id: str, intent: str = None):
    """
    Establece contexto de conversación para Sentry
    
    Args:
        conversation_id: ID de la conversación
        user_id: ID del usuario
        intent: Intención detectada (opcional)
    """
    
    sentry_sdk.set_tag("conversation_id", conversation_id)
    sentry_sdk.set_tag("user_id_hash", hash(user_id))  # Hash para privacidad
    
    if intent:
        sentry_sdk.set_tag("intent", intent)
    
    sentry_sdk.set_context("conversation", {
        "conversation_id": conversation_id,
        "intent": intent,
        "timestamp": str(timezone.now()) if 'timezone' in globals() else None
    })


def add_breadcrumb(message: str, category: str = 'custom', level: str = 'info', data: Dict[str, Any] = None):
    """
    Añade un breadcrumb a Sentry
    
    Args:
        message: Mensaje del breadcrumb
        category: Categoría del breadcrumb
        level: Nivel del breadcrumb
        data: Datos adicionales
    """
    
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {}
    )


class SentryMiddleware:
    """
    Middleware para añadir contexto automático a Sentry
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Establecer contexto de request
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("request_method", request.method)
            scope.set_tag("request_path", request.path)
            
            # Añadir información del usuario si está autenticado
            if hasattr(request, 'user') and request.user.is_authenticated:
                set_user_context(
                    user_id=str(request.user.id),
                    username=request.user.username,
                    email=request.user.email
                )
            
            # Añadir breadcrumb de request
            add_breadcrumb(
                message=f"{request.method} {request.path}",
                category="request",
                data={
                    "method": request.method,
                    "path": request.path,
                    "user_agent": request.META.get('HTTP_USER_AGENT', ''),
                }
            )
            
            response = self.get_response(request)
            
            # Añadir información de response
            scope.set_tag("response_status", response.status_code)
            
            return response


# Decorador para capturar errores en funciones específicas
def sentry_capture_errors(operation_name: str = None):
    """
    Decorador para capturar errores automáticamente en Sentry
    
    Args:
        operation_name: Nombre de la operación para contexto
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Añadir contexto de la función
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("function", func.__name__)
                    scope.set_tag("module", func.__module__)
                    
                    if operation_name:
                        scope.set_tag("operation", operation_name)
                    
                    scope.set_context("function_args", {
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    })
                    
                    capture_exception(e)
                
                # Re-raise la excepción
                raise
        
        return wrapper
    return decorator


# Configurar Sentry al importar el módulo si está disponible
if os.getenv('SENTRY_DSN'):
    configure_sentry()
else:
    logger.info("Sentry no configurado (SENTRY_DSN no encontrado)")
