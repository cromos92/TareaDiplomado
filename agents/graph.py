"""
Definición del grafo LangGraph para Farmacia IA
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AppState, AgentConfig
from .nodes import (
    RouterAgent, SupervisorAgent, PharmacyAgent, MedicationAgent,
    SafetyAgent, ClarificationAgent, EmergencyAgent, ResponseGenerator
)


def create_farmacia_graph(config: AgentConfig = None) -> StateGraph:
    """
    Crea el grafo principal de Farmacia IA
    
    Args:
        config: Configuración de agentes
    
    Returns:
        Grafo configurado de LangGraph
    """
    
    if config is None:
        config = AgentConfig()
    
    # Crear instancias de agentes
    router = RouterAgent(config)
    supervisor = SupervisorAgent(config)
    pharmacy = PharmacyAgent(config)
    medication = MedicationAgent(config)
    safety = SafetyAgent(config)
    clarification = ClarificationAgent(config)
    emergency = EmergencyAgent(config)
    response_gen = ResponseGenerator(config)
    
    # Crear el grafo
    workflow = StateGraph(AppState)
    
    # Añadir nodos
    workflow.add_node("router", router)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("pharmacy", pharmacy)
    workflow.add_node("medication", medication)
    workflow.add_node("safety", safety)
    workflow.add_node("clarification", clarification)
    workflow.add_node("emergency", emergency)
    workflow.add_node("response_generator", response_gen)
    
    # Definir punto de entrada
    workflow.set_entry_point("router")
    
    # Definir transiciones condicionales
    workflow.add_conditional_edges(
        "router",
        _route_after_classification,
        {
            "supervisor": "supervisor",
            "safety": "safety"
        }
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervision,
        {
            "pharmacy": "pharmacy",
            "medication": "medication", 
            "emergency": "emergency",
            "clarification": "clarification",
            "safety": "safety"
        }
    )
    
    workflow.add_conditional_edges(
        "safety",
        _route_after_safety,
        {
            "emergency": "emergency",
            "pharmacy": "pharmacy",
            "medication": "medication",
            "clarification": "clarification",
            "response_generator": "response_generator"
        }
    )
    
    # Transiciones condicionales desde medication (para manejar consultas mixtas)
    workflow.add_conditional_edges(
        "medication",
        _route_after_medication,
        {
            "pharmacy": "pharmacy",
            "response_generator": "response_generator"
        }
    )
    
    # Transiciones directas a generador de respuesta
    workflow.add_edge("pharmacy", "response_generator")
    workflow.add_edge("clarification", "response_generator")
    workflow.add_edge("emergency", "response_generator")
    
    # Fin del flujo
    workflow.add_edge("response_generator", END)
    
    return workflow


def _route_after_classification(state: AppState) -> Literal["supervisor", "safety"]:
    """
    Determina el siguiente paso después de la clasificación inicial
    """
    # Siempre pasar por safety primero para evaluación
    return "safety"


def _route_after_supervision(state: AppState) -> Literal["pharmacy", "medication", "emergency", "clarification", "safety"]:
    """
    Determina el agente especializado después de la supervisión
    """
    intent = state["intent"]
    slots = state["slots"]
    
    # Verificar si necesita clarificación
    if slots.get("needs_clarification", False):
        return "clarification"
    
    # Enrutar según intención
    routing_map = {
        "emergencia": "emergency",
        "medicamento": "medication",
        "farmacia": "pharmacy",
        "mixta": "medication",  # Empezar con medicamento para consultas mixtas
        "desconocida": "clarification"
    }
    
    return routing_map.get(intent, "clarification")


def _route_after_safety(state: AppState) -> Literal["emergency", "pharmacy", "medication", "clarification", "response_generator"]:
    """
    Determina el siguiente paso después de la evaluación de seguridad
    """
    # Verificar banderas críticas
    critical_flags = [
        flag for flag in state["safety_flags"]
        if flag["severity"] == "critical" and flag["action_required"]
    ]
    
    if critical_flags:
        return "emergency"
    
    # Si no hay banderas críticas, continuar con el flujo normal
    intent = state["intent"]
    slots = state["slots"]
    
    # Verificar si necesita clarificación
    if slots.get("needs_clarification", False) or intent == "desconocida":
        return "clarification"
    
    # Enrutar según intención
    routing_map = {
        "emergencia": "emergency",
        "medicamento": "medication", 
        "farmacia": "pharmacy",
        "mixta": "medication"  # Empezar con medicamento, luego pharmacy
    }
    
    return routing_map.get(intent, "response_generator")


def _route_after_medication(state: AppState) -> Literal["pharmacy", "response_generator"]:
    """
    Determina el siguiente paso después del MedicationAgent
    Si la intención es mixta, continúa con PharmacyAgent
    """
    intent = state["intent"]
    
    # Si es consulta mixta, continuar con pharmacy
    if intent == "mixta":
        return "pharmacy"
    
    # Para consultas solo de medicamento, ir directo a respuesta
    return "response_generator"


class FarmaciaGraphRunner:
    """
    Runner para ejecutar el grafo de Farmacia IA
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.graph = create_farmacia_graph(self.config)
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    async def process_message(
        self, 
        user_id: str, 
        message: str, 
        conversation_id: str = None,
        location: str = None
    ) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario a través del grafo
        
        Args:
            user_id: ID del usuario
            message: Mensaje del usuario
            conversation_id: ID de la conversación (opcional)
            location: Ubicación del usuario (opcional)
        
        Returns:
            Resultado del procesamiento con respuesta y metadatos
        """
        
        from .state import create_initial_state
        import uuid
        
        # Crear estado inicial
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        initial_state = create_initial_state(
            user_id=user_id,
            initial_message=message,
            trace_id=trace_id
        )
        
        # Configurar thread para memoria
        thread_config = {
            "configurable": {
                "thread_id": conversation_id or f"conv_{user_id}_{uuid.uuid4().hex[:8]}"
            }
        }
        
        try:
            # Ejecutar el grafo
            result = await self.app.ainvoke(initial_state, config=thread_config)
            
            # Extraer respuesta del asistente
            assistant_messages = [
                msg for msg in result["messages"]
                if msg["role"] == "assistant"
            ]
            
            response_text = ""
            if assistant_messages:
                response_text = "\n\n".join([msg["content"] for msg in assistant_messages])
            
            return {
                "success": True,
                "response": response_text,
                "trace_id": result["trace_id"],
                "intent": result["intent"],
                "safety_flags": result["safety_flags"],
                "conversation_id": thread_config["configurable"]["thread_id"],
                "metadata": {
                    "slots": result["slots"],
                    "message_count": len(result["messages"]),
                    "agents_used": self._extract_agents_used(result["messages"])
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id,
                "response": "Lo siento, ocurrió un error procesando tu consulta. Por favor, intenta nuevamente.",
                "conversation_id": thread_config["configurable"]["thread_id"]
            }
    
    def _extract_agents_used(self, messages: list) -> list:
        """Extrae los agentes que participaron en la conversación"""
        agents = set()
        
        for message in messages:
            if message["role"] == "assistant":
                metadata = message.get("metadata", {})
                agent = metadata.get("agent")
                if agent:
                    agents.add(agent)
        
        return list(agents)
    
    async def get_conversation_history(
        self, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Obtiene el historial de una conversación
        
        Args:
            conversation_id: ID de la conversación
        
        Returns:
            Historial de la conversación
        """
        
        thread_config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        try:
            # Obtener estado actual de la conversación
            state = await self.app.aget_state(config=thread_config)
            
            if state and state.values:
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "messages": state.values.get("messages", []),
                    "intent": state.values.get("intent"),
                    "safety_flags": state.values.get("safety_flags", [])
                }
            else:
                return {
                    "success": False,
                    "error": "Conversación no encontrada",
                    "conversation_id": conversation_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "conversation_id": conversation_id
            }


# Instancia global del runner (singleton)
_graph_runner = None


def get_graph_runner(config: AgentConfig = None) -> FarmaciaGraphRunner:
    """
    Obtiene la instancia global del graph runner
    
    Args:
        config: Configuración de agentes (solo se usa en la primera llamada)
    
    Returns:
        Instancia del graph runner
    """
    global _graph_runner
    
    if _graph_runner is None:
        _graph_runner = FarmaciaGraphRunner(config)
    
    return _graph_runner
