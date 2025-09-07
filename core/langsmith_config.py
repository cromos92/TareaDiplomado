"""
Configuración de LangSmith para tracing y monitoreo de agentes.
"""
import os
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class LangSmithConfig:
    """Configuración centralizada para LangSmith."""
    
    def __init__(self):
        self.api_key = os.getenv('LANGCHAIN_API_KEY')
        self.endpoint = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
        self.project = os.getenv('LANGCHAIN_PROJECT', 'farmacia-ia-project')
        self.tracing_enabled = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'
        self.timeout = int(os.getenv('LANGCHAIN_TIMEOUT', '30'))
        
        # Configurar variables de entorno para LangChain
        if self.api_key and self.tracing_enabled:
            os.environ['LANGCHAIN_API_KEY'] = self.api_key
            os.environ['LANGCHAIN_ENDPOINT'] = self.endpoint
            os.environ['LANGCHAIN_PROJECT'] = self.project
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            
            logger.info(f"LangSmith configurado - Proyecto: {self.project}")
        else:
            logger.warning("LangSmith no configurado - Tracing deshabilitado")
    
    @property
    def is_enabled(self) -> bool:
        """Verifica si LangSmith está habilitado."""
        return bool(self.api_key and self.tracing_enabled)
    
    def get_run_config(self, 
                      run_name: Optional[str] = None,
                      tags: Optional[list] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera configuración para un run de LangSmith.
        
        Args:
            run_name: Nombre del run
            tags: Tags para el run
            metadata: Metadata adicional
            
        Returns:
            Configuración del run
        """
        if not self.is_enabled:
            return {}
            
        config = {
            "project_name": self.project,
        }
        
        if run_name:
            config["run_name"] = run_name
            
        if tags:
            config["tags"] = tags
            
        if metadata:
            config["metadata"] = metadata
            
        return {"configurable": config}

# Instancia global
langsmith_config = LangSmithConfig()

def trace_agent(agent_name: str, 
               tags: Optional[list] = None,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Decorador para tracing automático de agentes.
    
    Args:
        agent_name: Nombre del agente
        tags: Tags adicionales
        metadata: Metadata adicional
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_config.is_enabled:
                return func(*args, **kwargs)
            
            try:
                from langsmith import traceable
                
                # Configurar tags por defecto
                default_tags = ["farmacia-ia", "agent", agent_name]
                if tags:
                    default_tags.extend(tags)
                
                # Configurar metadata por defecto
                default_metadata = {
                    "agent_name": agent_name,
                    "function": func.__name__,
                }
                if metadata:
                    default_metadata.update(metadata)
                
                # Aplicar tracing
                traced_func = traceable(
                    name=f"{agent_name}_{func.__name__}",
                    tags=default_tags,
                    metadata=default_metadata
                )(func)
                
                return traced_func(*args, **kwargs)
                
            except ImportError:
                logger.warning("langsmith no disponible - ejecutando sin tracing")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en tracing de {agent_name}: {e}")
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

def trace_conversation(conversation_id: str, user_id: str):
    """
    Decorador para tracing de conversaciones completas.
    
    Args:
        conversation_id: ID de la conversación
        user_id: ID del usuario
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not langsmith_config.is_enabled:
                return func(*args, **kwargs)
            
            try:
                from langsmith import traceable
                
                traced_func = traceable(
                    name=f"conversation_{func.__name__}",
                    tags=["farmacia-ia", "conversation"],
                    metadata={
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "function": func.__name__,
                    }
                )(func)
                
                return traced_func(*args, **kwargs)
                
            except ImportError:
                logger.warning("langsmith no disponible - ejecutando sin tracing")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en tracing de conversación: {e}")
                return func(*args, **kwargs)
                
        return wrapper
    return decorator

def log_agent_metrics(agent_name: str, 
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any],
                     execution_time: float,
                     success: bool = True,
                     error: Optional[str] = None):
    """
    Registra métricas de agentes en LangSmith.
    
    Args:
        agent_name: Nombre del agente
        input_data: Datos de entrada
        output_data: Datos de salida
        execution_time: Tiempo de ejecución en segundos
        success: Si la ejecución fue exitosa
        error: Mensaje de error si aplica
    """
    if not langsmith_config.is_enabled:
        return
    
    try:
        from langsmith import Client
        
        client = Client()
        
        # Crear run manual
        run = client.create_run(
            name=f"{agent_name}_execution",
            project_name=langsmith_config.project,
            run_type="llm",
            inputs=input_data,
            outputs=output_data,
            tags=["farmacia-ia", "agent", agent_name],
            extra={
                "execution_time": execution_time,
                "success": success,
                "error": error,
                "agent_name": agent_name,
            }
        )
        
        # Finalizar run solo si se creó correctamente
        if run and hasattr(run, 'id') and run.id:
            if success:
                client.update_run(run.id, end_time=run.start_time + execution_time)
            else:
                client.update_run(
                    run.id, 
                    end_time=run.start_time + execution_time,
                    error=error
                )
            
    except Exception as e:
        logger.error(f"Error registrando métricas en LangSmith: {e}")

def create_langsmith_session(session_id: str, user_id: str) -> Optional[str]:
    """
    Crea una sesión en LangSmith para agrupar runs relacionados.
    
    Args:
        session_id: ID de la sesión
        user_id: ID del usuario
        
    Returns:
        ID de la sesión creada o None si falla
    """
    if not langsmith_config.is_enabled:
        return None
    
    try:
        from langsmith import Client
        
        client = Client()
        
        session = client.create_session(
            name=f"farmacia_session_{session_id}",
            description=f"Sesión de farmacia IA para usuario {user_id}",
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "project": "farmacia-ia"
            }
        )
        
        return session.id
        
    except Exception as e:
        logger.error(f"Error creando sesión LangSmith: {e}")
        return None

def get_langsmith_url(run_id: str) -> Optional[str]:
    """
    Genera URL de LangSmith para un run específico.
    
    Args:
        run_id: ID del run
        
    Returns:
        URL del run en LangSmith
    """
    if not langsmith_config.is_enabled:
        return None
    
    base_url = langsmith_config.endpoint.replace('api.', '')
    return f"{base_url}/o/{langsmith_config.project}/runs/{run_id}"
