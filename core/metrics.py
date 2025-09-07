"""
Sistema de métricas para Farmacia IA usando Prometheus
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, REGISTRY
)
from django.http import HttpResponse
from django.conf import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Métricas de la Aplicación
# ============================================================================

# Contadores
http_requests_total = Counter(
    'farmacia_ia_http_requests_total',
    'Total de requests HTTP',
    ['method', 'endpoint', 'status_code']
)

api_requests_total = Counter(
    'farmacia_ia_api_requests_total', 
    'Total de requests a la API',
    ['endpoint', 'status_code']
)

chat_messages_total = Counter(
    'farmacia_ia_chat_messages_total',
    'Total de mensajes de chat procesados',
    ['intent', 'agent']
)

errors_total = Counter(
    'farmacia_ia_errors_total',
    'Total de errores por tipo',
    ['error_type', 'component']
)

# Histogramas (para medir duración)
http_request_duration = Histogram(
    'farmacia_ia_http_request_duration_seconds',
    'Duración de requests HTTP',
    ['method', 'endpoint']
)

api_request_duration = Histogram(
    'farmacia_ia_api_request_duration_seconds',
    'Duración de requests API',
    ['endpoint']
)

chat_processing_duration = Histogram(
    'farmacia_ia_chat_processing_duration_seconds',
    'Tiempo de procesamiento de chat',
    ['intent', 'agent']
)

database_query_duration = Histogram(
    'farmacia_ia_database_query_duration_seconds',
    'Duración de queries a la base de datos',
    ['operation', 'model']
)

# Gauges (para valores actuales)
active_connections = Gauge(
    'farmacia_ia_active_connections',
    'Conexiones activas'
)

active_conversations = Gauge(
    'farmacia_ia_active_conversations',
    'Conversaciones activas'
)

cache_hit_rate = Gauge(
    'farmacia_ia_cache_hit_rate',
    'Tasa de aciertos del cache'
)

memory_usage = Gauge(
    'farmacia_ia_memory_usage_bytes',
    'Uso de memoria en bytes'
)

# Info (para metadatos)
app_info = Info(
    'farmacia_ia_app_info',
    'Información de la aplicación'
)

# Inicializar info de la aplicación
app_info.info({
    'version': getattr(settings, 'VERSION', '1.0.0'),
    'environment': getattr(settings, 'ENVIRONMENT', 'development'),
    'django_version': getattr(settings, 'DJANGO_VERSION', 'unknown')
})


# ============================================================================
# Decoradores para Métricas
# ============================================================================

def track_http_requests(view_func):
    """Decorator para trackear requests HTTP"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        start_time = time.time()
        
        try:
            response = view_func(request, *args, **kwargs)
            status_code = str(response.status_code)
            
        except Exception as e:
            status_code = '500'
            errors_total.labels(
                error_type=type(e).__name__,
                component='django_view'
            ).inc()
            raise
        
        finally:
            # Registrar métricas
            duration = time.time() - start_time
            endpoint = request.resolver_match.url_name if request.resolver_match else 'unknown'
            
            http_requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            http_request_duration.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
        
        return response
    
    return wrapper


def track_api_requests(endpoint_name: str):
    """Decorator para trackear requests de API"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = '200'
            
            try:
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                status_code = '500'
                errors_total.labels(
                    error_type=type(e).__name__,
                    component='fastapi'
                ).inc()
                raise
            
            finally:
                duration = time.time() - start_time
                
                api_requests_total.labels(
                    endpoint=endpoint_name,
                    status_code=status_code
                ).inc()
                
                api_request_duration.labels(
                    endpoint=endpoint_name
                ).observe(duration)
        
        return wrapper
    return decorator


def track_chat_processing(intent: str = None, agent: str = None):
    """Decorator para trackear procesamiento de chat"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Intentar extraer intent y agent del estado si no se proporcionan
            actual_intent = intent
            actual_agent = agent
            
            if args and hasattr(args[0], 'get'):
                state = args[0]
                actual_intent = actual_intent or state.get('intent', 'unknown')
                actual_agent = actual_agent or func.__name__.replace('Agent', '').lower()
            
            try:
                result = func(*args, **kwargs)
                
                # Registrar mensaje procesado
                chat_messages_total.labels(
                    intent=actual_intent or 'unknown',
                    agent=actual_agent or 'unknown'
                ).inc()
                
                return result
                
            except Exception as e:
                errors_total.labels(
                    error_type=type(e).__name__,
                    component='agent'
                ).inc()
                raise
            
            finally:
                duration = time.time() - start_time
                chat_processing_duration.labels(
                    intent=actual_intent or 'unknown',
                    agent=actual_agent or 'unknown'
                ).observe(duration)
        
        return wrapper
    return decorator


@contextmanager
def track_database_query(operation: str, model: str):
    """Context manager para trackear queries de base de datos"""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        database_query_duration.labels(
            operation=operation,
            model=model
        ).observe(duration)


# ============================================================================
# Colector de Métricas del Sistema
# ============================================================================

class SystemMetricsCollector:
    """Colector de métricas del sistema"""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_cache_metrics(self, hit: bool):
        """Actualiza métricas de cache"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = self.cache_hits / total
            cache_hit_rate.set(hit_rate)
    
    def update_memory_usage(self):
        """Actualiza métricas de memoria"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage.set(memory_info.rss)
        except ImportError:
            logger.warning("psutil no disponible para métricas de memoria")
        except Exception as e:
            logger.error(f"Error obteniendo métricas de memoria: {e}")
    
    def update_active_conversations(self):
        """Actualiza contador de conversaciones activas"""
        try:
            from conversations.models import Conversation
            count = Conversation.objects.filter(closed_at__isnull=True).count()
            active_conversations.set(count)
        except Exception as e:
            logger.error(f"Error obteniendo conversaciones activas: {e}")


# Instancia global del colector
system_metrics = SystemMetricsCollector()


# ============================================================================
# Middleware de Métricas
# ============================================================================

class MetricsMiddleware:
    """Middleware para recopilar métricas automáticamente"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Incrementar conexiones activas
        active_connections.inc()
        
        start_time = time.time()
        
        try:
            response = self.get_response(request)
            status_code = str(response.status_code)
            
        except Exception as e:
            status_code = '500'
            errors_total.labels(
                error_type=type(e).__name__,
                component='middleware'
            ).inc()
            raise
        
        finally:
            # Decrementar conexiones activas
            active_connections.dec()
            
            # Registrar métricas si no es el endpoint de métricas
            if not request.path.startswith('/metrics'):
                duration = time.time() - start_time
                endpoint = getattr(request.resolver_match, 'url_name', 'unknown') if hasattr(request, 'resolver_match') else 'unknown'
                
                http_requests_total.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code=status_code
                ).inc()
                
                http_request_duration.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(duration)
        
        return response


# ============================================================================
# Vistas de Métricas
# ============================================================================

def metrics_view(request):
    """Vista para exponer métricas de Prometheus"""
    # Actualizar métricas del sistema antes de exportar
    system_metrics.update_memory_usage()
    system_metrics.update_active_conversations()
    
    # Generar métricas en formato Prometheus
    metrics_data = generate_latest(REGISTRY)
    
    return HttpResponse(
        metrics_data,
        content_type=CONTENT_TYPE_LATEST
    )


def health_metrics_view(request):
    """Vista con métricas de salud en formato JSON"""
    try:
        # Obtener métricas básicas
        from conversations.models import Conversation, Message
        
        total_conversations = Conversation.objects.count()
        active_conversations_count = Conversation.objects.filter(closed_at__isnull=True).count()
        total_messages = Message.objects.count()
        
        # Métricas de cache
        cache_hit_rate_value = cache_hit_rate._value._value if hasattr(cache_hit_rate._value, '_value') else 0
        
        metrics = {
            'status': 'healthy',
            'timestamp': time.time(),
            'conversations': {
                'total': total_conversations,
                'active': active_conversations_count,
            },
            'messages': {
                'total': total_messages,
            },
            'performance': {
                'cache_hit_rate': cache_hit_rate_value,
                'memory_usage_mb': memory_usage._value._value / (1024 * 1024) if hasattr(memory_usage._value, '_value') else 0,
            }
        }
        
        return HttpResponse(
            content=str(metrics).replace("'", '"'),
            content_type='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error generando métricas de salud: {e}")
        return HttpResponse(
            content='{"status": "error", "message": "Error generating metrics"}',
            content_type='application/json',
            status=500
        )


# ============================================================================
# Utilidades
# ============================================================================

def increment_error(error_type: str, component: str = 'unknown'):
    """Incrementa contador de errores"""
    errors_total.labels(
        error_type=error_type,
        component=component
    ).inc()


def record_chat_message(intent: str, agent: str):
    """Registra un mensaje de chat procesado"""
    chat_messages_total.labels(
        intent=intent,
        agent=agent
    ).inc()


def record_processing_time(duration: float, intent: str, agent: str):
    """Registra tiempo de procesamiento"""
    chat_processing_duration.labels(
        intent=intent,
        agent=agent
    ).observe(duration)


# ============================================================================
# Configuración de Métricas por Defecto
# ============================================================================

def setup_default_metrics():
    """Configura métricas por defecto"""
    # Registrar métricas iniciales
    app_info.info({
        'version': getattr(settings, 'VERSION', '1.0.0'),
        'environment': getattr(settings, 'ENVIRONMENT', 'development'),
        'started_at': str(time.time())
    })
    
    logger.info("Métricas de Prometheus configuradas correctamente")


# Configurar métricas al importar el módulo
setup_default_metrics()
