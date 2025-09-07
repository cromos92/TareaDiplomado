"""
Tests para middleware de conversations
"""

from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from unittest.mock import MagicMock
from .middleware import AuditMiddleware, RateLimitMiddleware


class AuditMiddlewareTest(TestCase):
    """Tests para AuditMiddleware"""
    
    def setUp(self):
        self.factory = RequestFactory()
        self.middleware = AuditMiddleware(lambda r: None)
    
    def test_get_user_id_authenticated(self):
        """Test obtener user_id de usuario autenticado"""
        user = User.objects.create_user('testuser', 'test@example.com', 'password')
        request = self.factory.get('/api/test')
        request.user = user
        
        user_id = self.middleware._get_user_id(request)
        self.assertEqual(user_id, str(user.id))
    
    def test_get_user_id_anonymous(self):
        """Test obtener user_id de usuario anónimo"""
        from django.contrib.auth.models import AnonymousUser
        
        request = self.factory.get('/api/test')
        request.user = AnonymousUser()
        
        # Mock session
        request.session = MagicMock()
        request.session.session_key = 'test_session_key'
        
        user_id = self.middleware._get_user_id(request)
        self.assertEqual(user_id, 'session:test_session_key')
    
    def test_get_client_ip(self):
        """Test obtener IP del cliente"""
        # IP directa
        request = self.factory.get('/api/test')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        ip = self.middleware._get_client_ip(request)
        self.assertEqual(ip, '192.168.1.100')
        
        # IP con proxy
        request.META['HTTP_X_FORWARDED_FOR'] = '203.0.113.1, 192.168.1.100'
        ip = self.middleware._get_client_ip(request)
        self.assertEqual(ip, '203.0.113.1')
    
    def test_should_audit(self):
        """Test determinar si debe auditarse"""
        # Rutas que deben auditarse
        audit_paths = ['/api/chat', '/admin/login', '/conversations/new']
        for path in audit_paths:
            request = self.factory.post(path)
            self.assertTrue(self.middleware._should_audit(request))
        
        # Rutas que no deben auditarse
        skip_paths = ['/static/css/style.css', '/media/image.jpg', '/favicon.ico']
        for path in skip_paths:
            request = self.factory.get(path)
            self.assertFalse(self.middleware._should_audit(request))


class RateLimitMiddlewareTest(TestCase):
    """Tests para RateLimitMiddleware"""
    
    def setUp(self):
        self.factory = RequestFactory()
        self.middleware = RateLimitMiddleware(lambda r: None)
    
    def test_should_rate_limit(self):
        """Test determinar si aplicar rate limiting"""
        # Rutas con rate limiting
        rate_limit_paths = ['/api/chat', '/api/conversations']
        for path in rate_limit_paths:
            request = self.factory.post(path)
            self.assertTrue(self.middleware._should_rate_limit(request))
        
        # Rutas sin rate limiting
        normal_paths = ['/admin/', '/conversations/', '/static/']
        for path in normal_paths:
            request = self.factory.get(path)
            self.assertFalse(self.middleware._should_rate_limit(request))
    
    def test_get_client_ip(self):
        """Test obtener IP del cliente"""
        request = self.factory.post('/api/chat')
        request.META['REMOTE_ADDR'] = '192.168.1.100'
        
        ip = self.middleware._get_client_ip(request)
        self.assertEqual(ip, '192.168.1.100')
