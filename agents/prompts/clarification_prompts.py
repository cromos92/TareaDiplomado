"""
Prompts y templates para el ClarificationAgent
"""

from typing import Dict, List, Any


class ClarificationPrompts:
    """Prompts específicos para solicitar clarificaciones"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un agente de clarificación especializado en obtener información específica.
    
    Tu función es:
    1. Identificar qué información falta para dar una respuesta útil
    2. Hacer preguntas específicas y dirigidas
    3. Guiar al usuario hacia consultas más precisas
    4. Proporcionar opciones claras cuando sea apropiado
    
    PRINCIPIOS:
    - Haz preguntas específicas, no generales
    - Ofrece opciones múltiples cuando sea útil
    - Mantén un tono amigable y profesional
    - Explica por qué necesitas la información adicional
    
    Responde de forma que facilite al usuario proporcionar la información necesaria.
    """
    
    # Template general de clarificación
    GENERAL_CLARIFICATION_TEMPLATE = """
    🤔 **Necesito más información para ayudarte mejor**
    
    Para brindarte la respuesta más útil y precisa, me gustaría saber:
    
    **❓ Preguntas específicas:**
    {questions}
    
    **💡 Esto me ayudará a:**
    - Darte información más precisa
    - Recomendarte las mejores opciones
    - Asegurarme de que sea seguro para ti
    
    ⚕️ *Mientras más específico seas, mejor te podré ayudar*
    """
    
    # Templates específicos por tipo de consulta
    MEDICATION_CLARIFICATION = """
    💊 **Clarificación sobre Medicamentos**
    
    Para ayudarte con información sobre medicamentos, necesito saber:
    
    **🔍 Información específica:**
    - ¿Qué medicamento específico te interesa?
    - ¿Es para ti o para otra persona?
    - ¿Qué síntomas o condición quieres tratar?
    - ¿Tomas actualmente otros medicamentos?
    - ¿Tienes alguna alergia conocida?
    
    **📋 También sería útil saber:**
    - Tu edad aproximada
    - Si estás embarazada o en lactancia
    - Si tienes condiciones médicas conocidas
    
    ⚕️ *Esta información me ayuda a darte recomendaciones más seguras*
    """
    
    PHARMACY_CLARIFICATION = """
    🏥 **Clarificación sobre Farmacias**
    
    Para ayudarte a encontrar la farmacia que necesitas:
    
    **📍 Ubicación:**
    - ¿En qué comuna o ciudad estás?
    - ¿Tienes una dirección de referencia?
    - ¿Qué tan lejos estás dispuesto a viajar?
    
    **🕒 Horario:**
    - ¿Necesitas farmacia ahora o puedes esperar?
    - ¿Buscas farmacias de turno específicamente?
    - ¿Qué días y horarios prefieres?
    
    **💊 Servicios:**
    - ¿Buscas algún medicamento específico?
    - ¿Necesitas servicios especiales? (inyecciones, presión arterial, etc.)
    
    ⚕️ *Con esta información te daré opciones más precisas*
    """
    
    SYMPTOM_CLARIFICATION = """
    🩺 **Clarificación sobre Síntomas**
    
    Para darte la mejor recomendación sobre tus síntomas:
    
    **🔍 Detalles del síntoma:**
    - ¿Cuándo comenzó?
    - ¿Qué tan intenso es del 1 al 10?
    - ¿Es constante o viene y va?
    - ¿Algo lo mejora o empeora?
    
    **📋 Contexto importante:**
    - ¿Has tenido esto antes?
    - ¿Tomas algún medicamento actualmente?
    - ¿Tienes condiciones médicas conocidas?
    - ¿Hay otros síntomas acompañantes?
    
    **⚠️ Signos de alarma:**
    Si tienes síntomas severos, fiebre alta, o dolor intenso, 
    considera buscar atención médica inmediata.
    
    ⚕️ *Mientras más detalles, mejor podré orientarte*
    """
    
    DOSAGE_CLARIFICATION = """
    📏 **Clarificación sobre Dosis**
    
    Para darte información precisa sobre dosis:
    
    **👤 Información personal:**
    - ¿Cuál es tu edad?
    - ¿Cuál es tu peso aproximado?
    - ¿Es para ti o para otra persona?
    
    **💊 Sobre el medicamento:**
    - ¿Qué medicamento específico?
    - ¿En qué presentación? (tabletas, jarabe, etc.)
    - ¿Has tomado este medicamento antes?
    
    **🏥 Historial médico:**
    - ¿Tienes problemas de hígado o riñones?
    - ¿Tomas otros medicamentos?
    - ¿Tienes alergias conocidas?
    
    ⚕️ *La dosis correcta depende de varios factores personales*
    """
    
    INTERACTION_CLARIFICATION = """
    ⚠️ **Clarificación sobre Interacciones**
    
    Para verificar posibles interacciones medicamentosas:
    
    **💊 Medicamentos actuales:**
    - ¿Qué medicamentos tomas regularmente?
    - ¿Incluye vitaminas o suplementos?
    - ¿Usas medicamentos naturales o hierbas?
    
    **🆕 Nuevo medicamento:**
    - ¿Qué medicamento quieres agregar?
    - ¿Es con o sin receta médica?
    - ¿Por cuánto tiempo lo planeas tomar?
    
    **🏥 Condiciones médicas:**
    - ¿Tienes condiciones médicas diagnosticadas?
    - ¿Problemas de hígado, riñón o corazón?
    
    ⚕️ *Las interacciones pueden ser peligrosas - mejor verificar*
    """
    
    EMERGENCY_CLARIFICATION = """
    🚨 **Clarificación de Urgencia**
    
    Para determinar si necesitas atención médica inmediata:
    
    **⚡ Síntomas actuales:**
    - ¿Qué síntomas tienes exactamente?
    - ¿Cuándo comenzaron?
    - ¿Están empeorando rápidamente?
    
    **🚨 Signos de alarma:**
    ¿Tienes alguno de estos síntomas?
    - Dificultad para respirar
    - Dolor de pecho intenso
    - Pérdida de conciencia
    - Sangrado abundante
    - Convulsiones
    
    **📞 Si tienes síntomas severos:**
    - Llama al 131 (SAMU) inmediatamente
    - Ve al hospital más cercano
    - No esperes a que empeore
    
    ⚕️ *Ante la duda en emergencias, siempre busca ayuda médica*
    """
    
    UNKNOWN_INTENT_TEMPLATE = """
    🤷‍♂️ **¿En qué puedo ayudarte?**
    
    No estoy seguro de entender exactamente qué necesitas. 
    
    **🏥 Puedo ayudarte con:**
    
    **💊 Medicamentos:**
    - Información sobre dosis y efectos
    - Recomendaciones para síntomas comunes
    - Interacciones medicamentosas
    
    **🏥 Farmacias:**
    - Ubicaciones y horarios
    - Farmacias de turno
    - Servicios disponibles
    
    **🩺 Consultas de salud:**
    - Síntomas menores
    - Primeros auxilios básicos
    - Cuándo buscar atención médica
    
    **❓ Por favor, dime específicamente:**
    - ¿Qué tipo de información necesitas?
    - ¿Es sobre un medicamento, farmacia, o síntoma?
    - ¿Es urgente o puedes esperar?
    
    ⚕️ *Estoy aquí para ayudarte con tus consultas farmacéuticas*
    """
    
    @classmethod
    def get_clarification_response(cls, clarification_type: str, **kwargs) -> str:
        """Obtiene respuesta de clarificación según el tipo"""
        
        clarification_map = {
            "medication": cls.MEDICATION_CLARIFICATION,
            "pharmacy": cls.PHARMACY_CLARIFICATION,
            "symptom": cls.SYMPTOM_CLARIFICATION,
            "dosage": cls.DOSAGE_CLARIFICATION,
            "interaction": cls.INTERACTION_CLARIFICATION,
            "emergency": cls.EMERGENCY_CLARIFICATION,
            "unknown": cls.UNKNOWN_INTENT_TEMPLATE,
            "general": cls.GENERAL_CLARIFICATION_TEMPLATE
        }
        
        template = clarification_map.get(clarification_type, cls.GENERAL_CLARIFICATION_TEMPLATE)
        
        if clarification_type == "general" and "questions" in kwargs:
            return template.format(**kwargs)
        else:
            return template
    
    @classmethod
    def determine_clarification_type(cls, message: str, intent: str, slots: Dict[str, Any]) -> str:
        """Determina qué tipo de clarificación se necesita"""
        
        message_lower = message.lower()
        
        # Si hay indicios de emergencia
        emergency_keywords = ["urgente", "grave", "dolor intenso", "emergencia"]
        if any(keyword in message_lower for keyword in emergency_keywords):
            return "emergency"
        
        # Basado en la intención detectada
        if intent == "medicamento":
            if any(word in message_lower for word in ["dosis", "cuanto", "cantidad"]):
                return "dosage"
            elif any(word in message_lower for word in ["interacción", "mezclar", "combinar"]):
                return "interaction"
            else:
                return "medication"
        elif intent == "farmacia":
            return "pharmacy"
        elif intent == "desconocida":
            return "unknown"
        else:
            return "general"
    
    @classmethod
    def generate_specific_questions(cls, missing_info: List[str]) -> List[str]:
        """Genera preguntas específicas basadas en información faltante"""
        
        question_map = {
            "medication_name": "¿Qué medicamento específico te interesa?",
            "symptoms": "¿Qué síntomas estás experimentando?",
            "age": "¿Cuál es tu edad aproximada?",
            "location": "¿En qué comuna o ciudad te encuentras?",
            "urgency": "¿Es urgente o puedes esperar?",
            "current_medications": "¿Tomas actualmente otros medicamentos?",
            "allergies": "¿Tienes alguna alergia conocida a medicamentos?",
            "medical_conditions": "¿Tienes alguna condición médica diagnosticada?",
            "pregnancy": "¿Estás embarazada o en período de lactancia?",
            "dosage_form": "¿En qué presentación prefieres el medicamento? (tabletas, jarabe, etc.)",
            "duration": "¿Por cuánto tiempo necesitas el tratamiento?"
        }
        
        return [question_map.get(info, f"Información sobre {info}") for info in missing_info]
    
    @classmethod
    def get_openai_prompt(cls, message: str, intent: str, missing_info: List[str]) -> Dict[str, str]:
        """Genera prompt para OpenAI para clarificaciones"""
        
        context = f"""
        Mensaje del usuario: {message}
        Intención detectada: {intent}
        Información faltante: {', '.join(missing_info)}
        """
        
        user_prompt = f"""
        {context}
        
        El usuario ha hecho una consulta pero falta información para dar una respuesta útil.
        
        Genera preguntas específicas y útiles para:
        1. Obtener la información faltante
        2. Clarificar la intención del usuario
        3. Asegurar que la respuesta sea segura y apropiada
        
        Mantén un tono amigable y explica por qué necesitas la información adicional.
        Ofrece opciones múltiples cuando sea apropiado.
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
