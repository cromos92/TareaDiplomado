"""
Nodos de agentes para el grafo LangGraph de Farmacia IA
"""

import re
import logging
import time
from typing import Dict, Any, List
from .state import AppState, add_message, add_safety_flag, update_slots, set_intent, AgentConfig
from .openai_client import OpenAIClient
from .prompts import (
    MedicationPrompts, PharmacyPrompts, SafetyPrompts, 
    EmergencyPrompts, ClarificationPrompts, ResponseGeneratorPrompts
)
from core.langsmith_config import trace_agent, log_agent_metrics, langsmith_config

logger = logging.getLogger(__name__)


class BaseAgent:
    """Clase base para todos los agentes"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.openai_client = OpenAIClient(config)
    
    def log_agent_action(self, agent_name: str, state: AppState, action: str, details: Dict[str, Any] = None):
        """Log de acciones del agente"""
        logger.info(
            f"Agent {agent_name} - {action}",
            extra={
                "trace_id": state["trace_id"],
                "user_id": state["user_id"],
                "intent": state["intent"],
                "details": details or {}
            }
        )


class RouterAgent(BaseAgent):
    """
    Agente enrutador que determina la intención del usuario
    y dirige el flujo hacia el agente especializado apropiado
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Procesa el estado y determina la intención"""
        
        start_time = time.time()
        self.log_agent_action("RouterAgent", state, "analyzing_intent")
        
        # Tracing manual para LangSmith
        if langsmith_config.is_enabled:
            try:
                from langsmith import traceable
                return self._process_with_tracing(state, start_time)
            except ImportError:
                logger.warning("langsmith no disponible")
        
        return self._process_intent(state, start_time)
    
    def _process_with_tracing(self, state: AppState, start_time: float) -> AppState:
        """Procesa con tracing de LangSmith"""
        from langsmith import traceable
        
        @traceable(
            name="RouterAgent_classify_intent",
            tags=["farmacia-ia", "router", "intent-classification"],
            metadata={
                "user_id": state["user_id"],
                "trace_id": state["trace_id"]
            }
        )
        def classify_with_trace(user_message: str) -> dict:
            intent = self._classify_intent(user_message)
            slots = self._extract_basic_slots(user_message)
            return {"intent": intent, "slots": slots}
        
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return set_intent(state, "desconocida")
        
        last_message = user_messages[-1]["content"].lower()
        
        # Ejecutar con tracing
        result = classify_with_trace(last_message)
        
        return self._update_state_with_result(state, result, start_time)
    
    def _process_intent(self, state: AppState, start_time: float) -> AppState:
        """Procesa sin tracing (fallback)"""
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return set_intent(state, "desconocida")
        
        last_message = user_messages[-1]["content"].lower()
        
        # Detectar intención basada en palabras clave y patrones
        intent = self._classify_intent(last_message)
        
        # Extraer slots básicos
        slots = self._extract_basic_slots(last_message)
        
        result = {"intent": intent, "slots": slots}
        return self._update_state_with_result(state, result, start_time)
    
    def _update_state_with_result(self, state: AppState, result: dict, start_time: float) -> AppState:
        """Actualiza el estado con el resultado y logs"""
        intent = result["intent"]
        slots = result["slots"]
        
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return set_intent(state, "desconocida")
        
        last_message = user_messages[-1]["content"].lower()
        
        # Actualizar estado
        new_state = set_intent(state, intent)
        new_state = update_slots(new_state, slots)
        
        self.log_agent_action(
            "RouterAgent", 
            new_state, 
            "intent_classified",
            {"detected_intent": intent, "extracted_slots": slots}
        )
        
        # Log métricas en LangSmith
        execution_time = time.time() - start_time
        log_agent_metrics(
            agent_name="RouterAgent",
            input_data={
                "user_message": last_message,
                "user_id": state["user_id"],
                "trace_id": state["trace_id"]
            },
            output_data={
                "detected_intent": intent,
                "extracted_slots": slots,
                "execution_time": execution_time
            },
            execution_time=execution_time,
            success=True
        )
        
        return new_state
    
    def _classify_intent(self, message: str) -> str:
        """Clasifica la intención basada en el contenido del mensaje"""
        
        # Palabras clave para emergencias
        emergency_keywords = [
            "emergencia", "urgente", "grave", "crítico", "dolor intenso",
            "no respira", "inconsciente", "sangrado", "alergia severa",
            "shock", "sobredosis", "intoxicación", "envenenamiento"
        ]
        
        # Palabras clave para medicamentos específicos
        medication_keywords = [
            "paracetamol", "ibuprofeno", "aspirina", "omeprazol", "losartán",
            "metformina", "atorvastatina", "amlodipino", "dosis", "mg",
            "comprimido", "cápsula", "jarabe", "gotas", "inyección",
            "dolor", "duele", "síntoma", "malestar", "acidez", "gastritis",
            "fiebre", "temperatura", "calentura", "cefalea", "migraña",
            "náuseas", "vómito", "diarrea", "estreñimiento", "tos",
            "tomar", "medicamento", "pastilla", "remedio", "tengo"
        ]
        
        # Palabras clave para consultas generales de farmacia
        pharmacy_keywords = [
            "farmacia", "receta", "prescripción", "genérico",
            "marca", "precio", "disponibilidad", "horario", "ubicación",
            "turno", "cerca", "dirección", "donde", "hay", "abierta",
            "antofagasta", "comuna", "ciudad"
        ]
        
        # Verificar emergencia (prioridad máxima)
        if any(keyword in message for keyword in emergency_keywords):
            return "emergencia"
        
        # Contar coincidencias para determinar intención
        medication_score = sum(1 for keyword in medication_keywords if keyword in message)
        pharmacy_score = sum(1 for keyword in pharmacy_keywords if keyword in message)
        
        # Si hay múltiples tipos de consulta
        if medication_score > 0 and pharmacy_score > 0:
            return "mixta"
        
        # Intención específica
        if medication_score > pharmacy_score:
            return "medicamento"
        elif pharmacy_score > 0:
            return "farmacia"
        
        return "desconocida"
    
    def _extract_basic_slots(self, message: str) -> Dict[str, Any]:
        """Extrae información básica del mensaje"""
        slots = {}
        
        # Extraer nombres de medicamentos comunes
        medications = [
            "paracetamol", "ibuprofeno", "aspirina", "omeprazol", "losartán",
            "metformina", "atorvastatina", "amlodipino", "simvastatina"
        ]
        
        found_medications = [med for med in medications if med in message.lower()]
        if found_medications:
            slots["medications"] = found_medications
        
        # Extraer dosis (números seguidos de mg, g, ml, etc.)
        dose_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|ui)'
        doses = re.findall(dose_pattern, message.lower())
        if doses:
            slots["doses"] = [f"{amount}{unit}" for amount, unit in doses]
        
        # Extraer edad si se menciona
        age_pattern = r'(\d+)\s*años?'
        age_match = re.search(age_pattern, message.lower())
        if age_match:
            slots["age"] = int(age_match.group(1))
        
        return slots


class SupervisorAgent(BaseAgent):
    """
    Agente supervisor que coordina el flujo entre agentes
    y toma decisiones de alto nivel
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Supervisa y coordina el flujo de agentes"""
        
        self.log_agent_action("SupervisorAgent", state, "supervising_flow")
        
        # Verificar si hay banderas de seguridad críticas
        critical_flags = [
            flag for flag in state["safety_flags"] 
            if flag["severity"] == "critical"
        ]
        
        if critical_flags:
            # Si hay banderas críticas, forzar flujo de emergencia
            new_state = set_intent(state, "emergencia")
            self.log_agent_action(
                "SupervisorAgent", 
                new_state, 
                "critical_safety_detected",
                {"critical_flags": len(critical_flags)}
            )
            return new_state
        
        # Verificar si necesita clarificación
        if state["intent"] == "desconocida" or not state["slots"]:
            # Marcar para clarificación
            slots_update = {"needs_clarification": True}
            new_state = update_slots(state, slots_update)
            
            self.log_agent_action(
                "SupervisorAgent", 
                new_state, 
                "clarification_needed"
            )
            return new_state
        
        # Flujo normal - continuar con el agente especializado
        self.log_agent_action(
            "SupervisorAgent", 
            state, 
            "routing_to_specialist",
            {"target_intent": state["intent"]}
        )
        
        return state


class PharmacyAgent(BaseAgent):
    """
    Agente especializado en consultas generales de farmacia
    """
    
    @trace_agent("PharmacyAgent", tags=["pharmacy-search", "location"])
    def __call__(self, state: AppState) -> AppState:
        """Procesa consultas generales de farmacia"""
        
        start_time = time.time()
        self.log_agent_action("PharmacyAgent", state, "processing_pharmacy_query")
        
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return state
        
        last_message = user_messages[-1]["content"]
        
        # TODO: Integración con OpenAI
        # response = self.openai_client.generate_pharmacy_response(last_message, state["slots"])
        
        # Placeholder response
        response = self._generate_pharmacy_response(last_message, state["slots"])
        
        # Añadir respuesta al estado
        new_state = add_message(state, "assistant", response, {
            "agent": "PharmacyAgent",
            "response_type": "pharmacy_general"
        })
        
        self.log_agent_action(
            "PharmacyAgent", 
            new_state, 
            "response_generated",
            {"response_length": len(response)}
        )
        
        return new_state
    
    def _generate_pharmacy_response(self, message: str, slots: Dict[str, Any]) -> str:
        """Genera respuesta sobre farmacias usando prompts organizados"""
        
        message_lower = message.lower()
        
        # Determinar tipo de consulta
        if "horario" in message_lower or "hora" in message_lower:
            return PharmacyPrompts.get_query_response("hours")
        elif "precio" in message_lower or "costo" in message_lower:
            return PharmacyPrompts.get_query_response("prices")
        elif "genérico" in message_lower:
            return PharmacyPrompts.get_query_response("generic")
        elif any(word in message_lower for word in ["ubicación", "donde", "cerca", "dirección"]):
            return PharmacyPrompts.get_query_response("locations")
        elif "receta" in message_lower or "prescripción" in message_lower:
            return PharmacyPrompts.get_query_response("prescription")
        else:
            return PharmacyPrompts.get_query_response("default")


class MedicationAgent(BaseAgent):
    """
    Agente especializado en consultas específicas sobre medicamentos
    """
    
    @trace_agent("MedicationAgent", tags=["medication-info", "dosage", "safety"])
    def __call__(self, state: AppState) -> AppState:
        """Procesa consultas específicas sobre medicamentos"""
        
        start_time = time.time()
        self.log_agent_action("MedicationAgent", state, "processing_medication_query")
        
        # Obtener medicamentos mencionados
        medications = state["slots"].get("medications", [])
        
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return state
        
        last_message = user_messages[-1]["content"]
        
        # TODO: Integración con OpenAI
        # response = self.openai_client.generate_medication_response(last_message, medications, state["slots"])
        
        # Placeholder response
        response = self._generate_medication_response(last_message, medications)
        
        # Añadir respuesta al estado
        new_state = add_message(state, "assistant", response, {
            "agent": "MedicationAgent",
            "response_type": "medication_specific",
            "medications_discussed": medications
        })
        
        self.log_agent_action(
            "MedicationAgent", 
            new_state, 
            "response_generated",
            {"medications_count": len(medications)}
        )
        
        return new_state
    
    def _generate_medication_response(self, message: str, medications: List[str]) -> str:
        """Genera respuesta específica sobre medicamentos usando prompts organizados"""
        
        message_lower = message.lower()
        
        # Detectar síntomas específicos usando los nuevos prompts
        symptoms = []
        if any(symptom in message_lower for symptom in ["dolor de estomago", "dolor de estómago", "duele el estomago", "duele el estómago", "dolor estomacal", "acidez", "gastritis"]):
            symptoms.append("dolor de estómago")
        elif any(symptom in message_lower for symptom in ["dolor de cabeza", "cefalea", "migraña", "jaqueca"]):
            symptoms.append("dolor de cabeza")
        elif any(symptom in message_lower for symptom in ["fiebre", "temperatura", "calentura"]):
            symptoms.append("fiebre")
        
        # Si hay síntomas detectados, usar template específico
        if symptoms:
            return MedicationPrompts.get_symptom_response(symptoms)
        
        # Si no hay medicamentos específicos, usar template por defecto
        if not medications:
            return MedicationPrompts.DEFAULT_MEDICATION_TEMPLATE
        
        # Si hay medicamentos específicos, usar información detallada
        medication = medications[0]
        return MedicationPrompts.get_medication_info(medication)


class SafetyAgent(BaseAgent):
    """
    Agente de seguridad que detecta y maneja situaciones de riesgo
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Evalúa la seguridad de la consulta usando prompts organizados"""
        
        self.log_agent_action("SafetyAgent", state, "evaluating_safety")
        
        # Obtener el último mensaje del usuario
        user_messages = [msg for msg in state["messages"] if msg["role"] == "user"]
        if not user_messages:
            return state
        
        last_message = user_messages[-1]["content"]
        medications = state["slots"].get("medications", [])
        
        # Usar el nuevo sistema de evaluación de seguridad
        safety_evaluation = SafetyPrompts.evaluate_safety_risk(
            message=last_message,
            medications=medications,
            user_profile={}  # TODO: Integrar perfil de usuario
        )
        
        # Añadir flags de seguridad al estado
        for flag in safety_evaluation["safety_flags"]:
            new_state = add_safety_flag(
                new_state,
                flag["type"],
                flag["severity"],
                flag["message"],
                flag["action_required"]
            )
        
        # Si hay riesgo alto o crítico, añadir respuesta de seguridad
        if safety_evaluation["risk_level"] in ["critical", "high"]:
            safety_response = SafetyPrompts.get_safety_response(safety_evaluation)
            new_state = add_message(new_state, "assistant", safety_response, {
                "agent": "SafetyAgent",
                "response_type": "safety_warning",
                "risk_level": safety_evaluation["risk_level"]
            })
        
        safety_flags_added = len(new_state["safety_flags"]) - len(state["safety_flags"])
        
        self.log_agent_action(
            "SafetyAgent", 
            new_state, 
            "safety_evaluation_complete",
            {"flags_added": safety_flags_added}
        )
        
        return new_state
    
    def _detect_emergency_signals(self, message: str) -> List[Dict[str, Any]]:
        """Detecta señales de emergencia en el mensaje"""
        flags = []
        
        critical_keywords = [
            "no respira", "inconsciente", "shock", "sobredosis",
            "intoxicación", "envenenamiento", "alergia severa"
        ]
        
        high_keywords = [
            "emergencia", "urgente", "grave", "crítico",
            "dolor intenso", "sangrado"
        ]
        
        for keyword in critical_keywords:
            if keyword in message:
                flags.append({
                    "type": "emergency",
                    "severity": "critical",
                    "message": f"Detectada situación crítica: {keyword}",
                    "action_required": True
                })
        
        for keyword in high_keywords:
            if keyword in message:
                flags.append({
                    "type": "emergency",
                    "severity": "high",
                    "message": f"Detectada situación urgente: {keyword}",
                    "action_required": True
                })
        
        return flags
    
    def _detect_dangerous_interactions(self, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta interacciones medicamentosas peligrosas"""
        flags = []
        
        medications = slots.get("medications", [])
        
        # Interacciones conocidas peligrosas (simplificado)
        dangerous_combinations = [
            (["aspirina", "warfarina"], "Riesgo de sangrado severo"),
            (["paracetamol", "alcohol"], "Riesgo de daño hepático"),
        ]
        
        for combination, warning in dangerous_combinations:
            if all(med in medications for med in combination):
                flags.append({
                    "type": "interaction_risk",
                    "severity": "high",
                    "message": f"Interacción peligrosa detectada: {warning}",
                    "action_required": True
                })
        
        return flags
    
    def _detect_dangerous_dosages(self, slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta dosis peligrosas"""
        flags = []
        
        doses = slots.get("doses", [])
        medications = slots.get("medications", [])
        
        # Límites de seguridad (simplificado)
        max_doses = {
            "paracetamol": "4000mg",
            "ibuprofeno": "2400mg",
            "aspirina": "4000mg"
        }
        
        for medication in medications:
            if medication in max_doses:
                for dose in doses:
                    # Extraer número de la dosis
                    import re
                    dose_match = re.search(r'(\d+)', dose)
                    if dose_match:
                        dose_amount = int(dose_match.group(1))
                        max_amount = int(re.search(r'(\d+)', max_doses[medication]).group(1))
                        
                        if dose_amount > max_amount:
                            flags.append({
                                "type": "dosage_warning",
                                "severity": "high",
                                "message": f"Dosis de {medication} excede límite seguro",
                                "action_required": True
                            })
        
        return flags


class ClarificationAgent(BaseAgent):
    """
    Agente que maneja solicitudes de clarificación
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Solicita clarificación al usuario"""
        
        self.log_agent_action("ClarificationAgent", state, "requesting_clarification")
        
        # Determinar qué tipo de clarificación se necesita
        clarification_message = self._generate_clarification_request(state)
        
        # Añadir mensaje de clarificación
        new_state = add_message(state, "assistant", clarification_message, {
            "agent": "ClarificationAgent",
            "response_type": "clarification_request"
        })
        
        self.log_agent_action(
            "ClarificationAgent", 
            new_state, 
            "clarification_sent"
        )
        
        return new_state
    
    def _generate_clarification_request(self, state: AppState) -> str:
        """Genera solicitud de clarificación apropiada"""
        
        if state["intent"] == "desconocida":
            return """
            🤔 **Necesito más información**
            
            Para ayudarte mejor, ¿podrías especificar qué tipo de consulta tienes?
            
            **Puedo ayudarte con**:
            - 💊 Información sobre medicamentos específicos
            - 🏥 Consultas generales de farmacia
            - ⚠️ Interacciones medicamentosas
            - 📋 Dosis y administración
            - 🕒 Horarios y ubicaciones
            
            ¿Cuál de estos temas te interesa más?
            """
        
        elif not state["slots"]:
            return """
            📝 **Información adicional necesaria**
            
            Para darte una respuesta precisa, necesito algunos detalles:
            
            - ¿Qué medicamento específico te interesa?
            - ¿Es sobre dosis, efectos secundarios, o interacciones?
            - ¿Hay alguna condición médica particular?
            
            Mientras más específico seas, mejor podré ayudarte.
            """
        
        else:
            return """
            🔍 **Aclarando tu consulta**
            
            Entiendo que tienes una consulta, pero me gustaría asegurarme de darte la información más útil.
            
            ¿Podrías reformular tu pregunta de manera más específica?
            
            Esto me ayudará a brindarte una respuesta más precisa y útil.
            """


class EmergencyAgent(BaseAgent):
    """
    Agente especializado en manejo de emergencias médicas
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Maneja situaciones de emergencia"""
        
        self.log_agent_action("EmergencyAgent", state, "handling_emergency")
        
        # Generar respuesta de emergencia
        emergency_response = self._generate_emergency_response(state)
        
        # Añadir respuesta de emergencia
        new_state = add_message(state, "assistant", emergency_response, {
            "agent": "EmergencyAgent",
            "response_type": "emergency_response",
            "priority": "critical"
        })
        
        self.log_agent_action(
            "EmergencyAgent", 
            new_state, 
            "emergency_response_sent"
        )
        
        return new_state
    
    def _generate_emergency_response(self, state: AppState) -> str:
        """Genera respuesta apropiada para emergencias"""
        
        critical_flags = [
            flag for flag in state["safety_flags"] 
            if flag["severity"] == "critical"
        ]
        
        if critical_flags:
            return """
            🚨 **EMERGENCIA MÉDICA DETECTADA**
            
            **ACCIÓN INMEDIATA REQUERIDA:**
            
            📞 **Llama inmediatamente a emergencias:**
            - Chile: 131 (SAMU)
            - España: 112
            - México: 911
            - Argentina: 107
            
            🏥 **Ve al hospital más cercano**
            
            ⚠️ **NO esperes, NO busques información online**
            
            **Esta es una situación que requiere atención médica profesional inmediata.**
            
            ---
            
            *Este sistema NO puede reemplazar la atención médica de emergencia.*
            """
        
        else:
            return """
            ⚠️ **Situación que requiere atención médica**
            
            Basándome en tu consulta, te recomiendo:
            
            1. 📞 **Contactar a tu médico** lo antes posible
            2. 🏥 **Acudir a urgencias** si es fuera de horario
            3. 📋 **Llevar lista de medicamentos** que tomas
            
            **Señales de alarma que requieren atención inmediata:**
            - Dificultad para respirar
            - Dolor en el pecho
            - Pérdida de conciencia
            - Sangrado abundante
            - Reacciones alérgicas severas
            
            ⚕️ *Ante la duda, siempre consulta con un profesional*
            """


class ResponseGenerator(BaseAgent):
    """
    Agente que genera la respuesta final consolidada
    """
    
    def __call__(self, state: AppState) -> AppState:
        """Genera la respuesta final del sistema"""
        
        self.log_agent_action("ResponseGenerator", state, "generating_final_response")
        
        # Verificar si ya hay una respuesta de agente especializado
        assistant_messages = [
            msg for msg in state["messages"] 
            if msg["role"] == "assistant" and msg.get("metadata", {}).get("agent") != "ResponseGenerator"
        ]
        
        if assistant_messages:
            # Ya hay respuesta, solo añadir disclaimer si es necesario
            if self.config.include_disclaimers:
                disclaimer = self._generate_disclaimer(state)
                new_state = add_message(state, "assistant", disclaimer, {
                    "agent": "ResponseGenerator",
                    "response_type": "disclaimer"
                })
            else:
                new_state = state
        else:
            # Generar respuesta por defecto
            default_response = self._generate_default_response(state)
            new_state = add_message(state, "assistant", default_response, {
                "agent": "ResponseGenerator",
                "response_type": "default"
            })
        
        self.log_agent_action(
            "ResponseGenerator", 
            new_state, 
            "final_response_ready"
        )
        
        return new_state
    
    def _generate_disclaimer(self, state: AppState) -> str:
        """Genera disclaimer apropiado"""
        
        if state["safety_flags"]:
            return """
            ---
            ⚕️ **IMPORTANTE**: Esta información es solo orientativa y no reemplaza la consulta médica profesional. Ante cualquier duda o síntoma, consulta con un médico o farmacéutico.
            """
        else:
            return """
            ---
            ℹ️ **Nota**: Esta información es orientativa. Para consejos personalizados, consulta con tu farmacéutico o médico.
            """
    
    def _generate_default_response(self, state: AppState) -> str:
        """Genera respuesta por defecto cuando no hay respuesta específica"""
        
        return """
        🏥 **Asistente de Farmacia IA**
        
        Hola, soy tu asistente especializado en consultas farmacéuticas.
        
        **Puedo ayudarte con**:
        - 💊 Información sobre medicamentos
        - 📋 Dosis y administración
        - ⚠️ Interacciones y contraindicaciones
        - 🏥 Consultas generales de farmacia
        - 🕒 Horarios y ubicaciones
        
        ¿En qué puedo ayudarte hoy?
        
        ⚕️ *Recuerda que esta información es orientativa y no reemplaza la consulta médica profesional.*
        """
