"""
Prompts y templates para el SafetyAgent
"""

from typing import Dict, List, Any


class SafetyPrompts:
    """Prompts específicos para evaluación de seguridad"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un agente de seguridad médica especializado en detectar situaciones de riesgo.
    
    Tu función es:
    1. Identificar síntomas o situaciones que requieren atención médica inmediata
    2. Detectar posibles interacciones medicamentosas peligrosas
    3. Evaluar si una consulta está fuera del alcance de automedicación
    4. Generar alertas apropiadas para situaciones de riesgo
    
    CRITERIOS DE SEGURIDAD:
    - Síntomas de emergencia médica
    - Medicamentos que requieren prescripción
    - Interacciones medicamentosas graves
    - Poblaciones vulnerables (embarazo, niños, ancianos)
    - Dosis peligrosas o uso inadecuado
    
    Responde con evaluaciones claras de riesgo y recomendaciones específicas.
    """
    
    # Palabras clave de emergencia
    EMERGENCY_KEYWORDS = [
        "emergencia", "urgente", "grave", "crítico", "dolor intenso",
        "no respira", "inconsciente", "sangrado", "alergia severa",
        "shock", "sobredosis", "intoxicación", "envenenamiento",
        "dolor de pecho", "dificultad respirar", "pérdida conciencia",
        "convulsiones", "vómito sangre", "fiebre muy alta"
    ]
    
    # Síntomas de alerta
    WARNING_SYMPTOMS = [
        "dolor pecho", "dificultad respirar", "mareos severos",
        "visión borrosa", "pérdida equilibrio", "confusión mental",
        "palpitaciones", "sudoración excesiva", "náuseas intensas",
        "dolor abdominal severo", "sangrado anormal", "erupción severa"
    ]
    
    # Medicamentos que requieren precaución especial
    HIGH_RISK_MEDICATIONS = [
        "warfarina", "insulina", "digoxina", "litio", "fenitoína",
        "carbamazepina", "metotrexato", "ciclosporina", "tacrolimus"
    ]
    
    # Templates de respuesta
    CRITICAL_SAFETY_ALERT = """
    🚨 **ALERTA DE SEGURIDAD CRÍTICA**
    
    Basándome en tu consulta, he detectado síntomas que requieren **atención médica inmediata**:
    
    **⚠️ Síntomas de alarma identificados:**
    {symptoms}
    
    **🏥 ACCIÓN REQUERIDA:**
    1. **Llama al 131 (SAMU)** inmediatamente
    2. **Acude a urgencias** del hospital más cercano
    3. **No te automediques** en esta situación
    
    **📞 Contactos de emergencia:**
    - SAMU: 131
    - Bomberos: 132
    - Carabineros: 133
    
    ⚕️ **Esta situación requiere evaluación médica profesional inmediata**
    """
    
    MODERATE_SAFETY_WARNING = """
    ⚠️ **Advertencia de Seguridad**
    
    Tu consulta presenta aspectos que requieren precaución:
    
    **🔍 Factores de riesgo identificados:**
    {risk_factors}
    
    **💡 Recomendaciones:**
    - Consulta con un médico antes de tomar medicamentos
    - No excedas las dosis recomendadas
    - Informa sobre otros medicamentos que tomas
    - Busca atención médica si los síntomas empeoran
    
    **📞 Si los síntomas se intensifican:**
    - Contacta a tu médico de cabecera
    - Acude a SAPU o servicio de urgencias
    
    ⚕️ *La automedicación puede ser riesgosa en tu situación*
    """
    
    MEDICATION_INTERACTION_WARNING = """
    ⚠️ **Posible Interacción Medicamentosa**
    
    He detectado una posible interacción entre medicamentos:
    
    **💊 Medicamentos involucrados:**
    {medications}
    
    **🚨 Riesgos potenciales:**
    - Efectos adversos aumentados
    - Reducción de eficacia
    - Toxicidad medicamentosa
    
    **📋 ACCIÓN REQUERIDA:**
    1. **Consulta con tu farmacéutico** antes de combinar
    2. **Informa a tu médico** sobre todos los medicamentos
    3. **No suspendas medicamentos** sin supervisión
    
    ⚕️ *Siempre verifica interacciones con un profesional*
    """
    
    PREGNANCY_SAFETY_WARNING = """
    🤱 **Precaución en Embarazo/Lactancia**
    
    Has mencionado estar embarazada o en período de lactancia.
    
    **⚠️ Consideraciones especiales:**
    - Muchos medicamentos están contraindicados
    - Algunos pueden afectar al bebé
    - Las dosis pueden necesitar ajuste
    
    **📋 RECOMENDACIÓN:**
    1. **Consulta con tu ginecólogo** antes de tomar cualquier medicamento
    2. **Informa siempre** tu estado a farmacéuticos y médicos
    3. **Usa solo medicamentos aprobados** para embarazo/lactancia
    
    **📞 Contactos útiles:**
    - Tu ginecólogo de cabecera
    - Matrona del consultorio
    - Servicio de urgencias obstétricas
    
    ⚕️ *La seguridad de tu bebé es prioritaria*
    """
    
    PEDIATRIC_SAFETY_WARNING = """
    👶 **Precaución en Niños**
    
    Has consultado sobre medicamentos para un menor de edad.
    
    **⚠️ Consideraciones pediátricas:**
    - Dosis diferentes según peso y edad
    - Algunos medicamentos están contraindicados
    - Mayor riesgo de efectos adversos
    
    **📋 RECOMENDACIÓN:**
    1. **Consulta con pediatra** antes de medicar
    2. **Nunca uses medicamentos de adultos** en niños
    3. **Verifica dosis según peso** del niño
    
    **🚨 Busca atención inmediata si el niño presenta:**
    - Fiebre muy alta (>39.5°C)
    - Dificultad para respirar
    - Vómitos persistentes
    - Letargo o irritabilidad extrema
    
    ⚕️ *Los niños requieren atención médica especializada*
    """
    
    ELDERLY_SAFETY_WARNING = """
    👴 **Precaución en Adultos Mayores**
    
    Has consultado sobre medicamentos para un adulto mayor.
    
    **⚠️ Consideraciones geriátricas:**
    - Mayor sensibilidad a medicamentos
    - Riesgo de interacciones aumentado
    - Metabolismo más lento
    - Múltiples medicamentos (polifarmacia)
    
    **📋 RECOMENDACIÓN:**
    1. **Consulta con geriatra** o médico de cabecera
    2. **Revisa todos los medicamentos** actuales
    3. **Ajusta dosis** según función renal/hepática
    
    **⚠️ Síntomas de alerta en adultos mayores:**
    - Confusión súbita
    - Caídas frecuentes
    - Pérdida de apetito
    - Cambios en el comportamiento
    
    ⚕️ *Los adultos mayores requieren supervisión médica estrecha*
    """
    
    NO_SAFETY_CONCERNS = """
    ✅ **Evaluación de Seguridad Completada**
    
    No se han detectado riesgos de seguridad inmediatos en tu consulta.
    
    **💡 Recordatorios generales:**
    - Sigue siempre las dosis recomendadas
    - Lee los prospectos de los medicamentos
    - Consulta con farmacéutico ante dudas
    - Busca atención médica si síntomas empeoran
    
    ⚕️ *Puedes proceder con las recomendaciones proporcionadas*
    """
    
    @classmethod
    def evaluate_safety_risk(cls, message: str, medications: List[str] = None, 
                           user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evalúa el riesgo de seguridad de una consulta"""
        
        message_lower = message.lower()
        medications = medications or []
        user_profile = user_profile or {}
        
        risk_level = "low"
        safety_flags = []
        recommendations = []
        
        # Verificar síntomas de emergencia
        emergency_detected = any(keyword in message_lower for keyword in cls.EMERGENCY_KEYWORDS)
        if emergency_detected:
            risk_level = "critical"
            safety_flags.append({
                "type": "emergency_symptoms",
                "severity": "critical",
                "message": "Síntomas de emergencia detectados",
                "action_required": True
            })
        
        # Verificar síntomas de alerta
        warning_symptoms = [symptom for symptom in cls.WARNING_SYMPTOMS if symptom in message_lower]
        if warning_symptoms and risk_level != "critical":
            risk_level = "moderate"
            safety_flags.append({
                "type": "warning_symptoms", 
                "severity": "moderate",
                "message": f"Síntomas de alerta: {', '.join(warning_symptoms)}",
                "action_required": True
            })
        
        # Verificar medicamentos de alto riesgo
        high_risk_meds = [med for med in medications if med.lower() in cls.HIGH_RISK_MEDICATIONS]
        if high_risk_meds:
            risk_level = "high" if risk_level == "low" else risk_level
            safety_flags.append({
                "type": "high_risk_medication",
                "severity": "high", 
                "message": f"Medicamentos de alto riesgo: {', '.join(high_risk_meds)}",
                "action_required": True
            })
        
        # Verificar poblaciones especiales
        if user_profile.get("is_pregnant") or "embarazo" in message_lower or "embarazada" in message_lower:
            risk_level = "moderate" if risk_level == "low" else risk_level
            safety_flags.append({
                "type": "pregnancy",
                "severity": "moderate",
                "message": "Consulta durante embarazo",
                "action_required": True
            })
        
        if user_profile.get("is_pediatric") or any(word in message_lower for word in ["niño", "niña", "bebé", "menor"]):
            risk_level = "moderate" if risk_level == "low" else risk_level
            safety_flags.append({
                "type": "pediatric",
                "severity": "moderate", 
                "message": "Consulta pediátrica",
                "action_required": True
            })
        
        if user_profile.get("is_elderly") or "adulto mayor" in message_lower:
            risk_level = "moderate" if risk_level == "low" else risk_level
            safety_flags.append({
                "type": "elderly",
                "severity": "moderate",
                "message": "Consulta geriátrica", 
                "action_required": True
            })
        
        return {
            "risk_level": risk_level,
            "safety_flags": safety_flags,
            "recommendations": recommendations,
            "requires_medical_attention": risk_level in ["critical", "high"]
        }
    
    @classmethod
    def get_safety_response(cls, safety_evaluation: Dict[str, Any]) -> str:
        """Genera respuesta basada en la evaluación de seguridad"""
        
        risk_level = safety_evaluation["risk_level"]
        safety_flags = safety_evaluation["safety_flags"]
        
        if risk_level == "critical":
            emergency_symptoms = [flag["message"] for flag in safety_flags if flag["type"] == "emergency_symptoms"]
            return cls.CRITICAL_SAFETY_ALERT.format(symptoms="\n".join(f"- {s}" for s in emergency_symptoms))
        
        elif risk_level in ["high", "moderate"]:
            # Determinar tipo específico de advertencia
            for flag in safety_flags:
                if flag["type"] == "pregnancy":
                    return cls.PREGNANCY_SAFETY_WARNING
                elif flag["type"] == "pediatric":
                    return cls.PEDIATRIC_SAFETY_WARNING
                elif flag["type"] == "elderly":
                    return cls.ELDERLY_SAFETY_WARNING
                elif flag["type"] == "high_risk_medication":
                    medications = [flag["message"] for flag in safety_flags if flag["type"] == "high_risk_medication"]
                    return cls.MEDICATION_INTERACTION_WARNING.format(medications="\n".join(f"- {m}" for m in medications))
            
            # Advertencia general moderada
            risk_factors = [flag["message"] for flag in safety_flags]
            return cls.MODERATE_SAFETY_WARNING.format(risk_factors="\n".join(f"- {r}" for r in risk_factors))
        
        else:
            return cls.NO_SAFETY_CONCERNS
    
    @classmethod
    def get_openai_prompt(cls, message: str, medications: List[str] = None) -> Dict[str, str]:
        """Genera prompt para OpenAI para evaluación de seguridad"""
        
        context = f"""
        Mensaje del usuario: {message}
        Medicamentos mencionados: {', '.join(medications) if medications else 'Ninguno'}
        """
        
        user_prompt = f"""
        {context}
        
        Evalúa esta consulta médica para identificar riesgos de seguridad:
        
        1. ¿Hay síntomas que requieren atención médica inmediata?
        2. ¿Se mencionan medicamentos que requieren precaución especial?
        3. ¿La consulta involucra poblaciones vulnerables (embarazo, niños, ancianos)?
        4. ¿Hay riesgo de interacciones medicamentosas?
        5. ¿La automedicación es apropiada en este caso?
        
        Proporciona una evaluación de riesgo clara y recomendaciones específicas.
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
