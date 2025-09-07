"""
Prompts y templates para el ResponseGenerator
"""

from typing import Dict, List, Any


class ResponseGeneratorPrompts:
    """Prompts para generar respuestas finales consolidadas"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un generador de respuestas finales para un asistente farmacéutico.
    
    Tu función es:
    1. Consolidar información de múltiples agentes especializados
    2. Generar respuestas coherentes y bien estructuradas
    3. Añadir disclaimers apropiados de seguridad
    4. Asegurar que la información sea clara y accionable
    
    PRINCIPIOS:
    - Combina información sin duplicar contenido
    - Mantén la estructura clara con secciones bien definidas
    - Incluye siempre disclaimers de seguridad apropiados
    - Prioriza la seguridad del usuario sobre todo
    
    Responde de forma profesional, clara y útil.
    """
    
    # Disclaimers por tipo de consulta
    MEDICATION_DISCLAIMER = """
    ⚕️ **Importante:** Esta información es orientativa y no reemplaza la consulta médica profesional. 
    Siempre consulta con un médico o farmacéutico antes de tomar medicamentos, especialmente si tienes 
    condiciones médicas preexistentes, estás embarazada, o tomas otros medicamentos.
    """
    
    PHARMACY_DISCLAIMER = """
    ℹ️ **Nota:** Los horarios y disponibilidad pueden cambiar. Te recomendamos llamar antes de 
    dirigirte a la farmacia para confirmar horarios y disponibilidad de medicamentos.
    """
    
    EMERGENCY_DISCLAIMER = """
    🚨 **Advertencia:** Esta información no reemplaza la atención médica de emergencia. 
    Ante cualquier duda sobre tu salud o síntomas graves, busca atención médica inmediata 
    llamando al 131 o acudiendo al hospital más cercano.
    """
    
    GENERAL_DISCLAIMER = """
    ⚕️ **Disclaimer:** La información proporcionada es solo para fines educativos y no constituye 
    consejo médico profesional. En caso de emergencia, llama al 131. Para consultas médicas 
    específicas, contacta a tu médico de cabecera.
    """
    
    # Templates para respuestas combinadas
    MIXED_MEDICATION_PHARMACY_TEMPLATE = """
    {medication_response}
    
    ---
    
    {pharmacy_response}
    
    ---
    
    **📋 Resumen de recomendaciones:**
    - Para tu síntoma: {medication_summary}
    - Dónde conseguirlo: {pharmacy_summary}
    
    {disclaimer}
    """
    
    SAFETY_WARNING_TEMPLATE = """
    {safety_warning}
    
    ---
    
    {original_response}
    
    {disclaimer}
    """
    
    # Template por defecto cuando no hay respuesta específica
    DEFAULT_RESPONSE_TEMPLATE = """
    🏥 **Asistente de Farmacia IA**
    
    Hola, soy tu asistente farmacéutico virtual. Puedo ayudarte con:
    
    **💊 Medicamentos:**
    - Información sobre medicamentos de venta libre
    - Dosis recomendadas y precauciones
    - Efectos secundarios e interacciones
    
    **🏥 Farmacias:**
    - Ubicaciones y horarios
    - Farmacias de turno
    - Disponibilidad de medicamentos
    
    **🩺 Consultas de salud:**
    - Síntomas menores comunes
    - Cuándo buscar atención médica
    - Primeros auxilios básicos
    
    **❓ ¿En qué puedo ayudarte hoy?**
    
    {disclaimer}
    """
    
    # Template para cuando hay múltiples agentes involucrados
    MULTI_AGENT_RESPONSE_TEMPLATE = """
    {primary_response}
    
    {secondary_responses}
    
    **📋 Información adicional:**
    {additional_info}
    
    {disclaimer}
    """
    
    @classmethod
    def get_appropriate_disclaimer(cls, response_type: str, safety_level: str = "low") -> str:
        """Obtiene el disclaimer apropiado según el tipo de respuesta"""
        
        if safety_level in ["critical", "high"]:
            return cls.EMERGENCY_DISCLAIMER
        elif response_type == "medication":
            return cls.MEDICATION_DISCLAIMER
        elif response_type == "pharmacy":
            return cls.PHARMACY_DISCLAIMER
        else:
            return cls.GENERAL_DISCLAIMER
    
    @classmethod
    def combine_responses(cls, responses: List[Dict[str, Any]], safety_evaluation: Dict[str, Any] = None) -> str:
        """Combina respuestas de múltiples agentes"""
        
        if not responses:
            return cls.DEFAULT_RESPONSE_TEMPLATE.format(
                disclaimer=cls.GENERAL_DISCLAIMER
            )
        
        # Separar respuestas por tipo de agente
        medication_responses = [r for r in responses if r.get("agent") == "MedicationAgent"]
        pharmacy_responses = [r for r in responses if r.get("agent") == "PharmacyAgent"]
        safety_responses = [r for r in responses if r.get("agent") == "SafetyAgent"]
        emergency_responses = [r for r in responses if r.get("agent") == "EmergencyAgent"]
        clarification_responses = [r for r in responses if r.get("agent") == "ClarificationAgent"]
        
        # Si hay respuesta de emergencia, priorizarla
        if emergency_responses:
            return emergency_responses[0]["content"] + "\n\n" + cls.EMERGENCY_DISCLAIMER
        
        # Si hay advertencias de seguridad críticas
        if safety_evaluation and safety_evaluation.get("risk_level") in ["critical", "high"]:
            safety_content = safety_responses[0]["content"] if safety_responses else ""
            original_content = ""
            
            if medication_responses:
                original_content = medication_responses[0]["content"]
            elif pharmacy_responses:
                original_content = pharmacy_responses[0]["content"]
            
            return cls.SAFETY_WARNING_TEMPLATE.format(
                safety_warning=safety_content,
                original_response=original_content,
                disclaimer=cls.EMERGENCY_DISCLAIMER
            )
        
        # Si hay respuestas de medicamento y farmacia (consulta mixta)
        if medication_responses and pharmacy_responses:
            medication_content = medication_responses[0]["content"]
            pharmacy_content = pharmacy_responses[0]["content"]
            
            # Extraer resúmenes
            medication_summary = cls._extract_summary(medication_content, "medication")
            pharmacy_summary = cls._extract_summary(pharmacy_content, "pharmacy")
            
            return cls.MIXED_MEDICATION_PHARMACY_TEMPLATE.format(
                medication_response=medication_content,
                pharmacy_response=pharmacy_content,
                medication_summary=medication_summary,
                pharmacy_summary=pharmacy_summary,
                disclaimer=cls.MEDICATION_DISCLAIMER
            )
        
        # Si hay clarificaciones necesarias
        if clarification_responses:
            return clarification_responses[0]["content"] + "\n\n" + cls.GENERAL_DISCLAIMER
        
        # Respuesta única
        if len(responses) == 1:
            response = responses[0]
            response_type = "medication" if response.get("agent") == "MedicationAgent" else "pharmacy"
            disclaimer = cls.get_appropriate_disclaimer(response_type)
            return response["content"] + "\n\n" + disclaimer
        
        # Múltiples respuestas - combinar
        primary_response = responses[0]["content"]
        secondary_responses = "\n\n".join([r["content"] for r in responses[1:]])
        
        return cls.MULTI_AGENT_RESPONSE_TEMPLATE.format(
            primary_response=primary_response,
            secondary_responses=secondary_responses,
            additional_info="Consulta con profesionales de la salud para información específica.",
            disclaimer=cls.GENERAL_DISCLAIMER
        )
    
    @classmethod
    def _extract_summary(cls, content: str, response_type: str) -> str:
        """Extrae un resumen breve del contenido"""
        
        if response_type == "medication":
            # Buscar recomendaciones de medicamentos
            if "omeprazol" in content.lower():
                return "Omeprazol para acidez, antiácidos para alivio rápido"
            elif "paracetamol" in content.lower():
                return "Paracetamol para dolor y fiebre"
            elif "ibuprofeno" in content.lower():
                return "Ibuprofeno para dolor e inflamación"
            else:
                return "Ver recomendaciones específicas arriba"
        
        elif response_type == "pharmacy":
            # Buscar información de ubicación
            if "antofagasta" in content.lower():
                return "Farmacias disponibles en Antofagasta"
            elif "turno" in content.lower():
                return "Farmacias de turno disponibles"
            else:
                return "Ver ubicaciones y horarios arriba"
        
        return "Ver información detallada arriba"
    
    @classmethod
    def add_safety_warnings(cls, content: str, safety_flags: List[Dict[str, Any]]) -> str:
        """Añade advertencias de seguridad al contenido"""
        
        if not safety_flags:
            return content
        
        critical_flags = [f for f in safety_flags if f.get("severity") == "critical"]
        moderate_flags = [f for f in safety_flags if f.get("severity") == "moderate"]
        
        warnings = []
        
        if critical_flags:
            warnings.append("🚨 **ATENCIÓN CRÍTICA:** Esta situación requiere atención médica inmediata.")
        
        if moderate_flags:
            warnings.append("⚠️ **PRECAUCIÓN:** Consulta con un profesional de la salud.")
        
        if warnings:
            warning_text = "\n".join(warnings)
            return f"{warning_text}\n\n---\n\n{content}"
        
        return content
    
    @classmethod
    def format_final_response(cls, content: str, metadata: Dict[str, Any] = None) -> str:
        """Formatea la respuesta final con metadatos"""
        
        metadata = metadata or {}
        
        # Añadir timestamp si está disponible
        if metadata.get("timestamp"):
            timestamp_note = f"\n\n*Información generada: {metadata['timestamp']}*"
            content += timestamp_note
        
        # Añadir información de fuentes si está disponible
        if metadata.get("sources"):
            sources = metadata["sources"]
            if "minsal" in sources:
                content += "\n\n🌐 *Datos de farmacias obtenidos de MINSAL*"
            if "openai" in sources:
                content += "\n\n🤖 *Respuesta generada con IA*"
        
        return content
    
    @classmethod
    def get_openai_prompt(cls, responses: List[Dict[str, Any]], user_query: str) -> Dict[str, str]:
        """Genera prompt para OpenAI para consolidar respuestas"""
        
        context = f"""
        Consulta del usuario: {user_query}
        
        Respuestas de agentes especializados:
        {chr(10).join([f"- {r.get('agent', 'Unknown')}: {r.get('content', '')[:200]}..." for r in responses])}
        """
        
        user_prompt = f"""
        {context}
        
        Consolida estas respuestas en una respuesta final coherente que:
        
        1. Combine la información sin duplicar contenido
        2. Mantenga una estructura clara y lógica
        3. Priorice la información más relevante para el usuario
        4. Incluya disclaimers de seguridad apropiados
        5. Sea fácil de leer y entender
        
        La respuesta debe ser completa pero concisa, y siempre priorizar la seguridad del usuario.
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
