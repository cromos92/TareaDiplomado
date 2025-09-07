# 🔍 Configuración de LangSmith para Farmacia IA

LangSmith es la plataforma de observabilidad y monitoreo de LangChain que permite tracear, debuggear y evaluar aplicaciones de IA. Esta guía explica cómo configurar LangSmith en el proyecto Farmacia IA.

## 📋 Requisitos Previos

1. **Cuenta de LangSmith**: Regístrate en [smith.langchain.com](https://smith.langchain.com)
2. **API Key**: Obtén tu API key desde el dashboard de LangSmith
3. **Proyecto**: Crea un proyecto en LangSmith (o usa el nombre por defecto)

## 🔧 Configuración

### 1. Variables de Entorno

Agrega las siguientes variables a tu archivo `.env`:

```bash
# LangSmith Configuration
LANGCHAIN_API_KEY=tu-api-key-aqui
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=farmacia-ia-project
LANGCHAIN_TRACING_V2=true
LANGCHAIN_TIMEOUT=30
```

### 2. Verificar Configuración

Ejecuta el comando de verificación:

```bash
python manage.py test_langsmith
```

Para ejecutar un test completo:

```bash
python manage.py test_langsmith --run-test
```

## 🎯 Funcionalidades Implementadas

### 1. Tracing Automático de Agentes

Los siguientes agentes tienen tracing automático habilitado:

- **RouterAgent**: Clasificación de intenciones
- **PharmacyAgent**: Búsqueda de farmacias
- **MedicationAgent**: Información de medicamentos
- **SafetyAgent**: Verificaciones de seguridad
- **EmergencyAgent**: Manejo de emergencias

### 2. Métricas Personalizadas

Se registran métricas detalladas incluyendo:

- Tiempo de ejecución de cada agente
- Datos de entrada y salida
- Estado de éxito/error
- Metadatos de contexto (user_id, trace_id)

### 3. Sesiones de Conversación

Cada conversación de WebSocket crea una sesión en LangSmith para agrupar todas las interacciones relacionadas.

## 📊 Monitoreo y Análisis

### Dashboard de LangSmith

Una vez configurado, podrás ver en el dashboard:

1. **Traces**: Flujo completo de cada consulta
2. **Runs**: Ejecuciones individuales de agentes
3. **Sessions**: Conversaciones agrupadas por usuario
4. **Metrics**: Estadísticas de rendimiento y errores

### Información Trackeada

- **Input/Output**: Mensajes de entrada y respuestas generadas
- **Latencia**: Tiempo de respuesta de cada agente
- **Errores**: Fallos y excepciones capturadas
- **Metadatos**: Contexto adicional (intención detectada, slots extraídos)

## 🔍 Debugging y Optimización

### Identificar Cuellos de Botella

1. Revisa los tiempos de ejecución por agente
2. Identifica patrones en consultas lentas
3. Analiza la distribución de intenciones

### Mejorar Prompts

1. Analiza respuestas incorrectas en el dashboard
2. Compara diferentes versiones de prompts
3. Evalúa la consistencia de las respuestas

## 🚀 Uso en Producción

### Configuración Recomendada

```bash
# Producción
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=farmacia-ia-production
LANGCHAIN_TIMEOUT=60

# Desarrollo
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=farmacia-ia-development
LANGCHAIN_TIMEOUT=30

# Testing
LANGCHAIN_TRACING_V2=false  # Deshabilitado para tests
```

### Consideraciones de Rendimiento

- El tracing agrega ~10-50ms de latencia por request
- Los datos se envían de forma asíncrona para minimizar impacto
- Configura timeouts apropiados según tu infraestructura

## 🛠️ Troubleshooting

### Problemas Comunes

1. **API Key Inválida**
   ```
   Error: Authentication failed
   Solución: Verifica tu LANGCHAIN_API_KEY
   ```

2. **Proyecto No Encontrado**
   ```
   Error: Project not found
   Solución: Crea el proyecto en el dashboard o cambia LANGCHAIN_PROJECT
   ```

3. **Timeout de Conexión**
   ```
   Error: Connection timeout
   Solución: Aumenta LANGCHAIN_TIMEOUT o verifica conectividad
   ```

### Logs de Debug

Para habilitar logs detallados:

```python
import logging
logging.getLogger('langsmith').setLevel(logging.DEBUG)
```

## 📈 Métricas Personalizadas

### Agregar Métricas Propias

```python
from core.langsmith_config import log_agent_metrics

log_agent_metrics(
    agent_name="CustomAgent",
    input_data={"query": "consulta del usuario"},
    output_data={"response": "respuesta generada"},
    execution_time=0.5,
    success=True
)
```

### Crear Sesiones Personalizadas

```python
from core.langsmith_config import create_langsmith_session

session_id = create_langsmith_session(
    session_id="custom_session_123",
    user_id="user_456"
)
```

## 🔗 Enlaces Útiles

- [Documentación de LangSmith](https://docs.smith.langchain.com/)
- [Dashboard de LangSmith](https://smith.langchain.com)
- [Guía de Tracing](https://docs.smith.langchain.com/tracing)
- [API Reference](https://docs.smith.langchain.com/reference)

## 💡 Consejos Avanzados

1. **Usa tags consistentes** para facilitar filtrado
2. **Agrupa runs relacionados** en sesiones
3. **Configura alertas** para errores críticos
4. **Exporta datos** para análisis offline
5. **Integra con CI/CD** para tests automáticos

---

¡Con LangSmith configurado, tendrás visibilidad completa del comportamiento de tus agentes de IA! 🎉
