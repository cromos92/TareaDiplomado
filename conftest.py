"""
Configuración global de pytest y fixtures compartidas
"""

import os
import pytest
import asyncio
from datetime import datetime, date
from unittest.mock import Mock, AsyncMock, patch
from typing import Generator, AsyncGenerator

import django
from django.conf import settings
from django.test import Client, TransactionTestCase
from django.contrib.auth.models import User
from django.core.management import call_command
from django.db import transaction

import redis.asyncio as redis
from fastapi.testclient import TestClient

# Configurar Django para tests
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()


# ============================================================================
# Fixtures de Base de Datos
# ============================================================================

@pytest.fixture(scope="session")
def django_db_setup():
    """Configuración de base de datos para toda la sesión de tests"""
    settings.DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
    
    # Crear tablas
    call_command('migrate', '--run-syncdb', verbosity=0, interactive=False)


@pytest.fixture
def db_with_data(db):
    """Base de datos con datos de prueba"""
    # Crear usuarios de prueba
    admin_user = User.objects.create_superuser(
        username='admin',
        email='admin@test.com',
        password='testpass123'
    )
    
    regular_user = User.objects.create_user(
        username='testuser',
        email='user@test.com',
        password='testpass123'
    )
    
    return {
        'admin_user': admin_user,
        'regular_user': regular_user
    }


@pytest.fixture
def transactional_db():
    """Base de datos con soporte transaccional"""
    return TransactionTestCase()


# ============================================================================
# Fixtures de Redis
# ============================================================================

@pytest.fixture
def redis_client():
    """Cliente Redis mock para tests"""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.keys = AsyncMock(return_value=[])
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.aclose = AsyncMock()
    
    return mock_redis


@pytest.fixture
async def real_redis_client():
    """Cliente Redis real para tests de integración"""
    try:
        client = redis.from_url("redis://localhost:6379/15", decode_responses=True)
        await client.ping()
        
        # Limpiar base de datos de test
        await client.flushdb()
        
        yield client
        
        # Limpiar después del test
        await client.flushdb()
        await client.aclose()
        
    except Exception:
        pytest.skip("Redis no disponible para tests de integración")


# ============================================================================
# Fixtures de Clientes HTTP
# ============================================================================

@pytest.fixture
def django_client():
    """Cliente Django para tests"""
    return Client()


@pytest.fixture
def authenticated_client(django_client, db_with_data):
    """Cliente Django autenticado"""
    django_client.force_login(db_with_data['regular_user'])
    return django_client


@pytest.fixture
def admin_client(django_client, db_with_data):
    """Cliente Django con usuario admin"""
    django_client.force_login(db_with_data['admin_user'])
    return django_client


@pytest.fixture
def fastapi_client():
    """Cliente FastAPI para tests"""
    from api.main import app
    return TestClient(app)


# ============================================================================
# Fixtures de Modelos
# ============================================================================

@pytest.fixture
def sample_conversation(db_with_data):
    """Conversación de ejemplo"""
    from conversations.models import Conversation
    
    return Conversation.objects.create(
        user_id=str(db_with_data['regular_user'].id),
        ip_address="192.168.1.100",
        encrypted=False,
        meta={"test": True}
    )


@pytest.fixture
def sample_message(sample_conversation):
    """Mensaje de ejemplo"""
    from conversations.models import Message
    
    return Message.objects.create(
        conversation=sample_conversation,
        role='user',
        content='¿Cuál es la dosis de paracetamol?',
        tokens=10
    )


@pytest.fixture
def sample_audit_log(sample_conversation):
    """Log de auditoría de ejemplo"""
    from conversations.models import AuditLog
    
    return AuditLog.objects.create(
        event='conversation_started',
        actor='test_user_hash',
        conversation=sample_conversation,
        payload={'test': True},
        ip_address='192.168.1.xxx'
    )


# ============================================================================
# Fixtures de Estados de Agentes
# ============================================================================

@pytest.fixture
def sample_app_state():
    """Estado de aplicación de ejemplo para agentes"""
    from agents.state import create_initial_state
    
    return create_initial_state(
        user_id="test_user_123",
        initial_message="¿Cuál es la dosis de paracetamol?",
        trace_id="test_trace_123"
    )


@pytest.fixture
def agent_config():
    """Configuración de agente para tests"""
    from agents.state import AgentConfig
    
    return AgentConfig(
        openai_model="gpt-3.5-turbo",
        openai_temperature=0.0,
        max_tokens=100,
        enable_safety_checks=True
    )


# ============================================================================
# Fixtures de APIs Externas (Mocks)
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Cliente OpenAI mock"""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Respuesta de prueba del asistente"
                    )
                )
            ]
        )
    )
    return mock_client


@pytest.fixture
def mock_minsal_client():
    """Cliente MINSAL mock"""
    from data.minsal_client import Farmacia, MinsalResponse
    
    mock_client = AsyncMock()
    
    # Datos de prueba
    sample_farmacias = [
        Farmacia(
            id="1",
            nombre="Farmacia Test",
            direccion="Calle Test 123",
            comuna="Ñuñoa",
            region="Región Metropolitana",
            telefono="+56912345678",
            latitud=-33.4569,
            longitud=-70.6483
        )
    ]
    
    mock_client.get_locales.return_value = MinsalResponse(
        data=sample_farmacias,
        timestamp=datetime.utcnow()
    )
    
    mock_client.get_locales_turno.return_value = MinsalResponse(
        data=sample_farmacias,
        timestamp=datetime.utcnow()
    )
    
    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Cliente Qdrant mock"""
    mock_client = AsyncMock()
    
    # Mock de respuestas de búsqueda
    mock_client.search.return_value = [
        Mock(
            id="doc_1",
            score=0.95,
            payload={
                "content": "Información sobre paracetamol",
                "source": "vademecum",
                "medication": "paracetamol"
            }
        )
    ]
    
    mock_client.upsert.return_value = Mock(status="completed")
    
    return mock_client


# ============================================================================
# Fixtures de Datos de Prueba
# ============================================================================

@pytest.fixture
def sample_chat_request():
    """Request de chat de ejemplo"""
    return {
        "user_id": "test_user_123",
        "text": "¿Cuál es la dosis de paracetamol?",
        "location": "Chile"
    }


@pytest.fixture
def sample_vademecum_data():
    """Datos de vademécum de ejemplo"""
    return [
        {
            "medicamento": "Paracetamol",
            "principio_activo": "Acetaminofén",
            "dosis_adulto": "500-1000mg cada 6-8h",
            "dosis_maxima": "4g/día",
            "indicaciones": "Dolor leve a moderado, fiebre",
            "contraindicaciones": "Hipersensibilidad, insuficiencia hepática grave",
            "advertencias": "No exceder dosis máxima, cuidado con alcohol",
            "interacciones": "Warfarina, carbamazepina"
        },
        {
            "medicamento": "Ibuprofeno",
            "principio_activo": "Ibuprofeno",
            "dosis_adulto": "400-600mg cada 6-8h",
            "dosis_maxima": "2400mg/día",
            "indicaciones": "Dolor, inflamación, fiebre",
            "contraindicaciones": "Úlcera péptica, insuficiencia renal",
            "advertencias": "Tomar con alimentos, riesgo cardiovascular",
            "interacciones": "Warfarina, ACE inhibidores, diuréticos"
        }
    ]


# ============================================================================
# Fixtures de Configuración
# ============================================================================

@pytest.fixture
def test_settings():
    """Settings de prueba"""
    return {
        'SECRET_KEY': 'test-secret-key',
        'DEBUG': True,
        'ENCRYPTION_KEY': 'dGVzdF9lbmNyeXB0aW9uX2tleV8zMl9ieXRlcw==',
        'REDIS_URL': 'redis://localhost:6379/15',
        'DATABASE_URL': 'sqlite:///:memory:',
    }


@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db):
    """Habilita acceso a DB para todos los tests automáticamente"""
    pass


# ============================================================================
# Fixtures de Tiempo
# ============================================================================

@pytest.fixture
def freeze_time():
    """Congela el tiempo para tests determinísticos"""
    with patch('django.utils.timezone.now') as mock_now:
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0)
        yield mock_now


@pytest.fixture
def sample_date():
    """Fecha de ejemplo para tests"""
    return date(2024, 1, 15)


# ============================================================================
# Fixtures de Archivos
# ============================================================================

@pytest.fixture
def temp_csv_file(tmp_path, sample_vademecum_data):
    """Archivo CSV temporal con datos de vademécum"""
    import pandas as pd
    
    csv_file = tmp_path / "vademecum_test.csv"
    df = pd.DataFrame(sample_vademecum_data)
    df.to_csv(csv_file, index=False)
    
    return csv_file


# ============================================================================
# Fixtures de Logging
# ============================================================================

@pytest.fixture
def capture_logs(caplog):
    """Captura logs para verificación en tests"""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# ============================================================================
# Fixtures de Seguridad
# ============================================================================

@pytest.fixture
def mock_encryption():
    """Mock de funciones de encriptación"""
    with patch('security.encryption.encrypt_text') as mock_encrypt, \
         patch('security.encryption.decrypt_text') as mock_decrypt:
        
        mock_encrypt.return_value = "encrypted_content"
        mock_decrypt.return_value = "decrypted_content"
        
        yield {
            'encrypt': mock_encrypt,
            'decrypt': mock_decrypt
        }


# ============================================================================
# Event Loop para Tests Async
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop para tests asíncronos"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Limpieza automática después de cada test"""
    yield
    
    # Limpiar caches
    from django.core.cache import cache
    cache.clear()
    
    # Limpiar archivos temporales si existen
    import tempfile
    import shutil
    temp_dirs = []
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ============================================================================
# Markers personalizados
# ============================================================================

def pytest_configure(config):
    """Configuración personalizada de pytest"""
    config.addinivalue_line(
        "markers", 
        "slow: marca tests como lentos (usar -m 'not slow' para omitir)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers",
        "security: marca tests de seguridad"
    )
    config.addinivalue_line(
        "markers",
        "compliance: marca tests de cumplimiento legal"
    )


def pytest_collection_modifyitems(config, items):
    """Modifica la colección de tests"""
    # Agregar marker 'slow' a tests que toman mucho tiempo
    for item in items:
        if "integration" in item.keywords or "slow" in item.name:
            item.add_marker(pytest.mark.slow)
