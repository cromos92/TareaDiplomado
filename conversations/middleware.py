"""
Middleware de auditoría para Farmacia IA
"""

import time
import logging
from django.utils.deprecation import MiddlewareMixin
from django.contrib.auth.models import AnonymousUser
from security.encryption import hash_user_id, mask_ip_address

logger = logging.getLogger(__name__)


class AuditMiddleware(MiddlewareMixin):
    """
    Middleware que registra eventos de auditoría sin PII
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)
    
    def process_request(self, request):
        """Procesa la request entrante"""
        # Marcar tiempo de inicio
        request._audit_start_time = time.time()
        
        # Obtener información del usuario y IP
        request._audit_user_id = self._get_user_id(request)
        request._audit_ip = self._get_client_ip(request)
        
        return None
    
    def process_response(self, request, response):
        """Procesa la response saliente"""
        try:
            # Calcular tiempo de procesamiento
            start_time = getattr(request, '_audit_start_time', time.time())
            processing_time = time.time() - start_time
            
            # Solo auditar ciertas rutas
            if self._should_audit(request):
                self._log_request(request, response, processing_time)
        
        except Exception as e:
            logger.error(f"Error en AuditMiddleware: {e}")
        
        return response
    
    def process_exception(self, request, exception):
        """Procesa excepciones"""
        try:
            if self._should_audit(request):
                self._log_exception(request, exception)
        except Exception as e:
            logger.error(f"Error logging exception in AuditMiddleware: {e}")
        
        return None
    
    def _get_user_id(self, request):
        """Obtiene el ID del usuario de forma segura"""
        try:
            if hasattr(request, 'user') and not isinstance(request.user, AnonymousUser):
                return str(request.user.id)
            
            # Para usuarios anónimos, usar session key
            session_key = request.session.session_key
            if session_key:
                return f"session:{session_key}"
            
            return "anonymous"
        
        except Exception:
            return "unknown"
    
    def _get_client_ip(self, request):
        """Obtiene la IP del cliente considerando proxies"""
        try:
            # Verificar headers de proxy
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip = x_forwarded_for.split(',')[0].strip()
                return ip
            
            x_real_ip = request.META.get('HTTP_X_REAL_IP')
            if x_real_ip:
                return x_real_ip
            
            # IP directa
            return request.META.get('REMOTE_ADDR', 'unknown')
        
        except Exception:
            return 'unknown'
    
    def _should_audit(self, request):
        """Determina si la request debe ser auditada"""
        path = request.path
        
        # Auditar estas rutas
        audit_paths = [
            '/api/',
            '/admin/',
            '/chat/',
            '/conversations/',
        ]
        
        # No auditar estas rutas
        skip_paths = [
            '/static/',
            '/media/',
            '/favicon.ico',
            '/health/',
            '/ping/',
        ]
        
        # Verificar si debe saltarse
        for skip_path in skip_paths:
            if path.startswith(skip_path):
                return False
        
        # Verificar si debe auditarse
        for audit_path in audit_paths:
            if path.startswith(audit_path):
                return True
        
        # Auditar solo métodos importantes en otras rutas
        return request.method in ['POST', 'PUT', 'DELETE', 'PATCH']
    
    def _log_request(self, request, response, processing_time):
        """Registra la request en el log de auditoría"""
        try:
            from .models import AuditLog
            
            # Preparar datos sin PII
            user_id = getattr(request, '_audit_user_id', 'unknown')
            ip_address = getattr(request, '_audit_ip', 'unknown')
            
            # Hash del user_id
            hashed_user = hash_user_id(user_id)
            
            # Enmascarar IP
            masked_ip = mask_ip_address(ip_address)
            
            # Determinar evento basado en la ruta y método
            event = self._determine_event(request, response)
            
            # Payload con información no sensible
            payload = {
                'method': request.method,
                'path': request.path,
                'status_code': response.status_code,
                'processing_time_ms': round(processing_time * 1000, 2),
                'user_agent': request.META.get('HTTP_USER_AGENT', '')[:200],
                'content_type': response.get('Content-Type', ''),
            }
            
            # Agregar información adicional para APIs
            if request.path.startswith('/api/'):
                payload.update({
                    'api_endpoint': request.path,
                    'query_params': dict(request.GET) if request.GET else {},
                })
            
            # Crear log de auditoría
            AuditLog.objects.create(
                event=event,
                actor=hashed_user,
                payload=payload,
                ip_address=masked_ip
            )
        
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
    
    def _log_exception(self, request, exception):
        """Registra excepciones en el log de auditoría"""
        try:
            from .models import AuditLog
            
            user_id = getattr(request, '_audit_user_id', 'unknown')
            ip_address = getattr(request, '_audit_ip', 'unknown')
            
            hashed_user = hash_user_id(user_id)
            masked_ip = mask_ip_address(ip_address)
            
            payload = {
                'method': request.method,
                'path': request.path,
                'exception_type': exception.__class__.__name__,
                'exception_message': str(exception)[:500],  # Limitar longitud
                'user_agent': request.META.get('HTTP_USER_AGENT', '')[:200],
            }
            
            AuditLog.objects.create(
                event='error_occurred',
                actor=hashed_user,
                payload=payload,
                ip_address=masked_ip
            )
        
        except Exception as e:
            logger.error(f"Error logging exception: {e}")
    
    def _determine_event(self, request, response):
        """Determina el tipo de evento basado en la request y response"""
        path = request.path
        method = request.method
        status = response.status_code
        
        # Eventos específicos por ruta
        if path.startswith('/api/chat'):
            return 'api_call'
        elif path.startswith('/api/'):
            return 'api_call'
        elif path.startswith('/admin/'):
            if method == 'POST' and status < 400:
                return 'admin_action'
            return 'admin_access'
        elif 'login' in path:
            if status < 400:
                return 'user_login'
            return 'login_failed'
        elif 'logout' in path:
            return 'user_logout'
        
        # Eventos genéricos
        if status >= 400:
            return 'error_occurred'
        elif method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            return 'data_modification'
        
        return 'page_access'


class RateLimitMiddleware(MiddlewareMixin):
    """
    Middleware básico de rate limiting por IP
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)
    
    def process_request(self, request):
        """Verifica rate limiting"""
        try:
            # Solo aplicar rate limiting a ciertas rutas
            if not self._should_rate_limit(request):
                return None
            
            from django.core.cache import cache
            from django.http import JsonResponse
            from django.conf import settings
            
            # Obtener IP del cliente
            ip = self._get_client_ip(request)
            
            # Configuración de rate limiting
            max_requests = getattr(settings, 'RATE_LIMIT_REQUESTS', 100)
            window_seconds = getattr(settings, 'RATE_LIMIT_WINDOW', 3600)
            
            # Clave de cache
            cache_key = f"rate_limit:{ip}"
            
            # Obtener contador actual
            current_count = cache.get(cache_key, 0)
            
            if current_count >= max_requests:
                # Registrar evento de rate limiting
                self._log_rate_limit_exceeded(request, ip, current_count)
                
                return JsonResponse({
                    'error': 'Rate limit exceeded',
                    'detail': f'Maximum {max_requests} requests per hour allowed'
                }, status=429)
            
            # Incrementar contador
            cache.set(cache_key, current_count + 1, window_seconds)
            
            return None
        
        except Exception as e:
            logger.error(f"Error in RateLimitMiddleware: {e}")
            return None
    
    def _should_rate_limit(self, request):
        """Determina si aplicar rate limiting"""
        path = request.path
        
        # Aplicar rate limiting a APIs
        rate_limit_paths = [
            '/api/chat',
            '/api/conversations',
        ]
        
        return any(path.startswith(rl_path) for rl_path in rate_limit_paths)
    
    def _get_client_ip(self, request):
        """Obtiene la IP del cliente"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', 'unknown')
        return ip
    
    def _log_rate_limit_exceeded(self, request, ip, count):
        """Registra cuando se excede el rate limit"""
        try:
            from .models import AuditLog
            from security.encryption import mask_ip_address
            
            masked_ip = mask_ip_address(ip)
            
            payload = {
                'path': request.path,
                'method': request.method,
                'request_count': count,
                'user_agent': request.META.get('HTTP_USER_AGENT', '')[:200],
            }
            
            AuditLog.objects.create(
                event='rate_limit_exceeded',
                actor='rate_limiter',
                payload=payload,
                ip_address=masked_ip
            )
        
        except Exception as e:
            logger.error(f"Error logging rate limit: {e}")
