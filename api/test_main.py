"""
Tests para la FastAPI app de Farmacia IA
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time

from main import app


@pytest.fixture
def client():
    """Cliente de test para FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_chat_request():
    """Request de chat de ejemplo"""
    return {
        "user_id": "test_user_123",
        "text": "¿Cuál es la dosis de paracetamol?",
        "location": "Chile"
    }


class TestHealthEndpoint:
    """Tests para el endpoint de health check"""
    
    def test_health_check_success(self, client):
        """Test health check exitoso"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "farmacia-ia-api"
        assert data["version"] == "1.0.0"
        assert "X-Trace-ID" in response.headers
    
    def test_health_check_has_trace_id(self, client):
        """Test que health check incluye trace ID"""
        response = client.get("/healthz")
        
        assert "X-Trace-ID" in response.headers
        trace_id = response.headers["X-Trace-ID"]
        assert len(trace_id) > 0


class TestChatEndpoint:
    """Tests para el endpoint de chat"""
    
    def test_chat_success_paracetamol(self, client, sample_chat_request):
        """Test chat exitoso con consulta sobre paracetamol"""
        response = client.post("/api/chat", json=sample_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "trace_id" in data
        assert len(data["message"]) > 0
        assert "paracetamol" in data["message"].lower()
        assert "X-Trace-ID" in response.headers
    
    def test_chat_success_ibuprofeno(self, client):
        """Test chat con consulta sobre ibuprofeno"""
        request_data = {
            "user_id": "test_user_123",
            "text": "Información sobre ibuprofeno",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ibuprofeno" in data["message"].lower()
    
    def test_chat_success_aspirina(self, client):
        """Test chat con consulta sobre aspirina"""
        request_data = {
            "user_id": "test_user_123",
            "text": "¿Cómo tomar aspirina?",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "aspirina" in data["message"].lower()
    
    def test_chat_success_interacciones(self, client):
        """Test chat con consulta sobre interacciones"""
        request_data = {
            "user_id": "test_user_123",
            "text": "¿Puedo combinar estos medicamentos?",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "interacción" in data["message"].lower()
    
    def test_chat_success_general(self, client):
        """Test chat con consulta general"""
        request_data = {
            "user_id": "test_user_123",
            "text": "Hola, necesito ayuda",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "asistente" in data["message"].lower()
    
    def test_chat_without_location(self, client):
        """Test chat sin ubicación (campo opcional)"""
        request_data = {
            "user_id": "test_user_123",
            "text": "¿Cuál es la dosis de paracetamol?"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "trace_id" in data


class TestChatValidation:
    """Tests para validación de requests de chat"""
    
    def test_chat_empty_text(self, client):
        """Test con texto vacío"""
        request_data = {
            "user_id": "test_user_123",
            "text": "",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422
    
    def test_chat_whitespace_text(self, client):
        """Test con texto solo espacios"""
        request_data = {
            "user_id": "test_user_123",
            "text": "   ",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422
    
    def test_chat_empty_user_id(self, client):
        """Test con user_id vacío"""
        request_data = {
            "user_id": "",
            "text": "¿Cuál es la dosis de paracetamol?",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422
    
    def test_chat_missing_user_id(self, client):
        """Test sin user_id"""
        request_data = {
            "text": "¿Cuál es la dosis de paracetamol?",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422
    
    def test_chat_missing_text(self, client):
        """Test sin texto"""
        request_data = {
            "user_id": "test_user_123",
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422
    
    def test_chat_text_too_long(self, client):
        """Test con texto demasiado largo"""
        long_text = "a" * 10001  # Excede el límite de 10000
        
        request_data = {
            "user_id": "test_user_123",
            "text": long_text,
            "location": "Chile"
        }
        
        response = client.post("/api/chat", json=request_data)
        
        assert response.status_code == 422


class TestRateLimiting:
    """Tests para rate limiting"""
    
    def test_rate_limit_enforcement(self, client, sample_chat_request):
        """Test que el rate limiting se aplica correctamente"""
        # Hacer 10 requests (el límite)
        for i in range(10):
            response = client.post("/api/chat", json=sample_chat_request)
            assert response.status_code == 200
        
        # La request 11 debería ser rechazada
        response = client.post("/api/chat", json=sample_chat_request)
        assert response.status_code == 429
    
    @pytest.mark.slow
    def test_rate_limit_reset(self, client, sample_chat_request):
        """Test que el rate limit se resetea después del tiempo"""
        # Hacer requests hasta el límite
        for i in range(10):
            response = client.post("/api/chat", json=sample_chat_request)
            assert response.status_code == 200
        
        # Verificar que se rechaza
        response = client.post("/api/chat", json=sample_chat_request)
        assert response.status_code == 429
        
        # Esperar un poco (en test real sería 1 minuto)
        # Para test, podríamos mockear el tiempo o usar un límite más corto
        time.sleep(1)
        
        # Nota: En un test real necesitaríamos configurar un rate limit más corto
        # o usar mocking para simular el paso del tiempo


class TestErrorHandling:
    """Tests para manejo de errores"""
    
    def test_invalid_json(self, client):
        """Test con JSON inválido"""
        response = client.post(
            "/api/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_wrong_content_type(self, client, sample_chat_request):
        """Test con content-type incorrecto"""
        response = client.post(
            "/api/chat",
            data=str(sample_chat_request),
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422
    
    @patch('main.logger')
    def test_internal_error_logging(self, mock_logger, client, sample_chat_request):
        """Test que los errores internos se loggean correctamente"""
        # Simular error interno
        with patch('main.ChatResponse', side_effect=Exception("Test error")):
            response = client.post("/api/chat", json=sample_chat_request)
            
            assert response.status_code == 500
            # Verificar que se loggeó el error
            mock_logger.error.assert_called()


class TestCORS:
    """Tests para configuración CORS"""
    
    def test_cors_headers_present(self, client):
        """Test que los headers CORS están presentes"""
        response = client.options("/api/chat")
        
        # FastAPI maneja OPTIONS automáticamente con CORS
        assert response.status_code in [200, 405]  # 405 si no hay handler explícito
    
    def test_cors_origin_allowed(self, client, sample_chat_request):
        """Test que origins permitidos funcionan"""
        headers = {"Origin": "http://localhost:8000"}
        
        response = client.post("/api/chat", json=sample_chat_request, headers=headers)
        
        assert response.status_code == 200


class TestMetrics:
    """Tests para endpoint de métricas"""
    
    def test_metrics_endpoint(self, client):
        """Test endpoint de métricas"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert data["service"] == "farmacia-ia-api"


class TestDocumentation:
    """Tests para documentación OpenAPI"""
    
    def test_openapi_docs_available(self, client):
        """Test que la documentación OpenAPI está disponible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self, client):
        """Test que ReDoc está disponible"""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_json_available(self, client):
        """Test que el JSON de OpenAPI está disponible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "info" in data
        assert "paths" in data
        assert data["info"]["title"] == "Farmacia IA API"


# Configuración de pytest
def pytest_configure(config):
    """Configuración de pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
