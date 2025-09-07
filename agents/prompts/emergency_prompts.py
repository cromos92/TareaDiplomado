"""
Prompts y templates para el EmergencyAgent
"""

from typing import Dict, List, Any


class EmergencyPrompts:
    """Prompts específicos para situaciones de emergencia"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un agente de emergencias médicas especializado en situaciones críticas.
    
    Tu función es:
    1. Proporcionar instrucciones claras para emergencias médicas
    2. Dirigir a servicios de emergencia apropiados
    3. Dar primeros auxilios básicos cuando sea seguro
    4. Mantener la calma y proporcionar información vital
    
    PRIORIDADES:
    1. Seguridad del paciente
    2. Activación de servicios de emergencia
    3. Instrucciones de primeros auxilios
    4. Información para el personal médico
    
    NUNCA:
    - Reemplaces atención médica profesional
    - Des instrucciones que puedan causar daño
    - Minimices la gravedad de una emergencia
    
    Responde con urgencia, claridad y precisión.
    """
    
    # Template principal de emergencia
    GENERAL_EMERGENCY_TEMPLATE = """
    🚨 **EMERGENCIA MÉDICA DETECTADA**
    
    **⚡ ACCIÓN INMEDIATA REQUERIDA:**
    
    1. 📞 **LLAMA AL 131 (SAMU) AHORA**
    2. 🏥 **Dirígete al hospital más cercano**
    3. 🆘 **No intentes automedicarte**
    
    **📞 Números de emergencia:**
    - **SAMU: 131**
    - **Bomberos: 132** 
    - **Carabineros: 133**
    
    **💡 Mientras esperas ayuda:**
    - Mantén la calma
    - No dejes sola a la persona
    - Prepara documentos de identidad
    - Anota síntomas y hora de inicio
    
    ⚕️ **Esta situación requiere atención médica profesional inmediata**
    """
    
    # Templates específicos por tipo de emergencia
    CHEST_PAIN_EMERGENCY = """
    🚨 **DOLOR DE PECHO - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA** - Posible infarto
    2. 🏥 **Ve a urgencias inmediatamente**
    3. 💊 **Si tienes nitroglicerina prescrita, úsala**
    
    **💡 Mientras esperas:**
    - Siéntate o recuéstate cómodamente
    - Afloja ropa ajustada
    - Si tienes aspirina y no eres alérgico, mastica 1 tableta
    - Mantente despierto y alerta
    
    **🚨 NO hagas:**
    - Ejercicio o esfuerzo
    - Conducir al hospital
    - Ignorar el dolor
    
    ⚕️ **El dolor de pecho puede ser un infarto - actúa rápido**
    """
    
    BREATHING_DIFFICULTY_EMERGENCY = """
    🚨 **DIFICULTAD RESPIRATORIA - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA**
    2. 🏥 **Urgencias inmediatamente**
    3. 🪑 **Siéntate erguido, no te acuestes**
    
    **💡 Mientras esperas:**
    - Mantén posición sentada
    - Afloja ropa del cuello y pecho
    - Respira lento y profundo
    - Abre ventanas para aire fresco
    
    **🚨 Si hay alergia conocida:**
    - Usa autoinyector de epinefrina si lo tienes
    - Informa al 131 sobre la alergia
    
    ⚕️ **La dificultad respiratoria puede ser mortal - busca ayuda ya**
    """
    
    UNCONSCIOUSNESS_EMERGENCY = """
    🚨 **PERSONA INCONSCIENTE - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA**
    2. 🔍 **Verifica si respira**
    3. 🤲 **RCP si no respira**
    
    **💡 Pasos básicos:**
    - Inclina cabeza hacia atrás, levanta mentón
    - Mira, escucha y siente respiración
    - Si no respira: 30 compresiones + 2 respiraciones
    - Compresiones en centro del pecho, fuerte y rápido
    
    **🚨 Posición de recuperación si respira:**
    - De lado, cabeza inclinada
    - Vía aérea libre
    - Monitorea respiración constantemente
    
    ⚕️ **Cada minuto cuenta - actúa rápido pero con cuidado**
    """
    
    SEVERE_BLEEDING_EMERGENCY = """
    🚨 **SANGRADO SEVERO - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA**
    2. 🩸 **Presión directa sobre la herida**
    3. 🏥 **Prepárate para ir a urgencias**
    
    **💡 Control de sangrado:**
    - Presión firme y constante con tela limpia
    - Eleva la parte lesionada si es posible
    - NO remuevas objetos clavados
    - Añade más vendas si se empapan
    
    **🚨 Signos de shock:**
    - Piel pálida y fría
    - Pulso rápido y débil
    - Mareos o desmayo
    - Sed intensa
    
    ⚕️ **El sangrado severo puede ser mortal en minutos**
    """
    
    POISONING_EMERGENCY = """
    🚨 **INTOXICACIÓN - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA**
    2. 🧪 **Identifica la sustancia**
    3. 🏥 **Ve a urgencias con el envase**
    
    **💡 Información vital:**
    - ¿Qué sustancia?
    - ¿Cuánta cantidad?
    - ¿Cuándo ocurrió?
    - ¿Síntomas actuales?
    
    **🚨 NO hagas:**
    - Provocar vómito (puede empeorar)
    - Dar líquidos sin autorización médica
    - Dejar sola a la persona
    
    **📞 Centro de Información Toxicológica:**
    - CITUC: 2 635 3800
    
    ⚕️ **Lleva el envase del producto al hospital**
    """
    
    ALLERGIC_REACTION_EMERGENCY = """
    🚨 **REACCIÓN ALÉRGICA SEVERA - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 AHORA**
    2. 💉 **Usa autoinyector de epinefrina si tienes**
    3. 🏥 **Urgencias inmediatamente**
    
    **🚨 Signos de anafilaxia:**
    - Dificultad respiratoria
    - Hinchazón de cara/garganta
    - Erupción generalizada
    - Pulso rápido
    - Mareos o desmayo
    
    **💡 Mientras esperas:**
    - Mantén vías respiratorias libres
    - Posición cómoda (sentado si respira bien)
    - Afloja ropa ajustada
    - Monitorea signos vitales
    
    ⚕️ **La anafilaxia puede ser mortal en minutos**
    """
    
    SEIZURE_EMERGENCY = """
    🚨 **CONVULSIONES - EMERGENCIA**
    
    **⚡ ACCIÓN INMEDIATA:**
    1. 📞 **LLAMA AL 131 si dura >5 min**
    2. 🛡️ **Protege de lesiones**
    3. ⏰ **Cronometra duración**
    
    **💡 Durante la convulsión:**
    - NO pongas nada en la boca
    - Aparta objetos peligrosos
    - Coloca algo suave bajo la cabeza
    - Afloja ropa del cuello
    - Observa y cronometra
    
    **🚨 Llama 131 si:**
    - Primera convulsión
    - Dura más de 5 minutos
    - Convulsiones repetidas
    - Lesión durante convulsión
    - Embarazo o diabetes
    
    ⚕️ **Después: posición de recuperación, monitorea respiración**
    """
    
    # Template para situaciones que requieren atención pero no son críticas
    URGENT_CARE_TEMPLATE = """
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
    
    @classmethod
    def get_emergency_response(cls, emergency_type: str, **kwargs) -> str:
        """Obtiene respuesta específica según tipo de emergencia"""
        
        emergency_map = {
            "chest_pain": cls.CHEST_PAIN_EMERGENCY,
            "breathing_difficulty": cls.BREATHING_DIFFICULTY_EMERGENCY,
            "unconsciousness": cls.UNCONSCIOUSNESS_EMERGENCY,
            "severe_bleeding": cls.SEVERE_BLEEDING_EMERGENCY,
            "poisoning": cls.POISONING_EMERGENCY,
            "allergic_reaction": cls.ALLERGIC_REACTION_EMERGENCY,
            "seizure": cls.SEIZURE_EMERGENCY,
            "urgent_care": cls.URGENT_CARE_TEMPLATE,
            "general": cls.GENERAL_EMERGENCY_TEMPLATE
        }
        
        return emergency_map.get(emergency_type, cls.GENERAL_EMERGENCY_TEMPLATE)
    
    @classmethod
    def classify_emergency_type(cls, message: str, safety_flags: List[Dict] = None) -> str:
        """Clasifica el tipo de emergencia basado en síntomas"""
        
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["dolor pecho", "dolor de pecho", "infarto", "corazón"]):
            return "chest_pain"
        elif any(keyword in message_lower for keyword in ["no respira", "dificultad respirar", "ahogo", "asfixia"]):
            return "breathing_difficulty"
        elif any(keyword in message_lower for keyword in ["inconsciente", "desmayo", "no despierta"]):
            return "unconsciousness"
        elif any(keyword in message_lower for keyword in ["sangrado", "hemorragia", "sangre"]):
            return "severe_bleeding"
        elif any(keyword in message_lower for keyword in ["intoxicación", "envenenamiento", "sobredosis"]):
            return "poisoning"
        elif any(keyword in message_lower for keyword in ["alergia severa", "anafilaxia", "hinchazón"]):
            return "allergic_reaction"
        elif any(keyword in message_lower for keyword in ["convulsiones", "convulsión", "epilepsia"]):
            return "seizure"
        else:
            return "general"
    
    @classmethod
    def get_openai_prompt(cls, message: str, emergency_type: str = None) -> Dict[str, str]:
        """Genera prompt para OpenAI para situaciones de emergencia"""
        
        context = f"""
        Mensaje del usuario: {message}
        Tipo de emergencia detectada: {emergency_type or 'General'}
        """
        
        user_prompt = f"""
        {context}
        
        Esta es una situación de emergencia médica. Proporciona:
        
        1. Instrucciones claras e inmediatas
        2. Números de emergencia relevantes
        3. Pasos de primeros auxilios si es apropiado
        4. Qué NO hacer para evitar empeorar la situación
        5. Información que será útil para el personal médico
        
        Mantén un tono urgente pero calmado. La seguridad del paciente es la prioridad.
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
