"""
Definición del estado de la aplicación para LangGraph
"""

from typing import TypedDict, List, Dict, Any, Literal, Optional
from dataclasses import dataclass
import uuid


# Tipos de intención que puede detectar el sistema
IntentType = Literal[
    "farmacia",      # Consultas generales sobre farmacia
    "medicamento",   # Consultas específicas sobre medicamentos
    "emergencia",    # Situaciones de emergencia médica
    "mixta",         # Consultas que combinan múltiples intenciones
    "desconocida"    # Intención no clara o no clasificable
]


class Message(TypedDict):
    """Estructura de un mensaje en la conversación"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]]


class SafetyFlag(TypedDict):
    """Estructura de una bandera de seguridad"""
    type: Literal["emergency", "contraindication", "dosage_warning", "interaction_risk"]
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    action_required: bool


class AppState(TypedDict):
    """
    Estado principal de la aplicación LangGraph
    
    Contiene toda la información necesaria para procesar
    una consulta farmacéutica a través del grafo de agentes.
    """
    
    # Identificación del usuario
    user_id: str
    
    # Intención detectada de la consulta
    intent: IntentType
    
    # Slots extraídos de la consulta (entidades, parámetros)
    slots: Dict[str, Any]
    
    # Historial de mensajes de la conversación
    messages: List[Message]
    
    # Banderas de seguridad detectadas
    safety_flags: List[SafetyFlag]
    
    # ID de trazabilidad para logging
    trace_id: str


@dataclass
class AgentConfig:
    """Configuración para los agentes"""
    
    # OpenAI settings
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.1
    max_tokens: int = 1000
    
    # Safety settings
    enable_safety_checks: bool = True
    emergency_keywords: List[str] = None
    
    # Response settings
    max_response_length: int = 2000
    include_disclaimers: bool = True
    
    def __post_init__(self):
        if self.emergency_keywords is None:
            self.emergency_keywords = [
                "emergencia", "urgente", "grave", "crítico",
                "dolor intenso", "no respira", "inconsciente",
                "sangrado", "alergia severa", "shock anafiláctico",
                "sobredosis", "intoxicación", "envenenamiento"
            ]


def create_initial_state(
    user_id: str,
    initial_message: str,
    trace_id: Optional[str] = None
) -> AppState:
    """
    Crea el estado inicial para una nueva conversación
    
    Args:
        user_id: ID del usuario
        initial_message: Primer mensaje del usuario
        trace_id: ID de trazabilidad (se genera si no se proporciona)
    
    Returns:
        Estado inicial de la aplicación
    """
    if trace_id is None:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    
    return AppState(
        user_id=user_id,
        intent="desconocida",  # Se determinará en el RouterAgent
        slots={},
        messages=[
            Message(
                role="user",
                content=initial_message,
                timestamp=str(uuid.uuid4()),  # En producción usar timestamp real
                metadata={}
            )
        ],
        safety_flags=[],
        trace_id=trace_id
    )


def add_message(state: AppState, role: str, content: str, metadata: Dict[str, Any] = None) -> AppState:
    """
    Añade un nuevo mensaje al estado
    
    Args:
        state: Estado actual
        role: Rol del mensaje (user, assistant, system)
        content: Contenido del mensaje
        metadata: Metadatos adicionales
    
    Returns:
        Estado actualizado con el nuevo mensaje
    """
    new_message = Message(
        role=role,
        content=content,
        timestamp=str(uuid.uuid4()),  # En producción usar timestamp real
        metadata=metadata or {}
    )
    
    # Crear nuevo estado con el mensaje añadido
    new_state = state.copy()
    new_state["messages"] = state["messages"] + [new_message]
    
    return new_state


def add_safety_flag(
    state: AppState,
    flag_type: str,
    severity: str,
    message: str,
    action_required: bool = False
) -> AppState:
    """
    Añade una bandera de seguridad al estado
    
    Args:
        state: Estado actual
        flag_type: Tipo de bandera de seguridad
        severity: Severidad (low, medium, high, critical)
        message: Mensaje descriptivo
        action_required: Si requiere acción inmediata
    
    Returns:
        Estado actualizado con la nueva bandera
    """
    new_flag = SafetyFlag(
        type=flag_type,
        severity=severity,
        message=message,
        action_required=action_required
    )
    
    new_state = state.copy()
    new_state["safety_flags"] = state["safety_flags"] + [new_flag]
    
    return new_state


def update_slots(state: AppState, new_slots: Dict[str, Any]) -> AppState:
    """
    Actualiza los slots del estado
    
    Args:
        state: Estado actual
        new_slots: Nuevos slots para añadir/actualizar
    
    Returns:
        Estado actualizado con los nuevos slots
    """
    new_state = state.copy()
    updated_slots = state["slots"].copy()
    updated_slots.update(new_slots)
    new_state["slots"] = updated_slots
    
    return new_state


def set_intent(state: AppState, intent: IntentType) -> AppState:
    """
    Establece la intención del estado
    
    Args:
        state: Estado actual
        intent: Nueva intención
    
    Returns:
        Estado actualizado con la nueva intención
    """
    new_state = state.copy()
    new_state["intent"] = intent
    
    return new_state
