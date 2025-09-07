"""
Prompts y templates para el MedicationAgent
"""

from typing import Dict, List, Any


class MedicationPrompts:
    """Prompts específicos para consultas de medicamentos"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un asistente farmacéutico especializado en medicamentos. Tu función es:
    
    1. Proporcionar información precisa sobre medicamentos de venta libre
    2. Recomendar opciones seguras para síntomas comunes
    3. Identificar cuándo se requiere consulta médica
    4. Incluir siempre disclaimers de seguridad apropiados
    
    IMPORTANTE:
    - NO diagnostiques enfermedades
    - NO prescribas medicamentos con receta
    - SIEMPRE incluye advertencias de seguridad
    - Recomienda consulta médica cuando sea apropiado
    
    Responde de forma clara, estructurada y profesional.
    """
    
    # Templates para síntomas específicos
    STOMACH_PAIN_TEMPLATE = """
    🩺 **Dolor de Estómago - Recomendaciones**
    
    Para dolor estomacal leve, puedes considerar:
    
    **💊 Opciones de venta libre:**
    - **Omeprazol** (20mg): Para acidez y gastritis
    - **Ranitidina** (150mg): Reductor de ácido  
    - **Antiácidos** (Alka-Seltzer, Mylanta): Alivio rápido
    - **Simeticona**: Para gases y distensión
    
    **🚨 Consulta médico urgente si tienes:**
    - Dolor intenso o persistente
    - Vómitos con sangre
    - Heces negras o con sangre
    - Fiebre alta
    - Pérdida de peso inexplicable
    
    **💡 Consejos adicionales:**
    - Evita alimentos irritantes (picantes, grasosos)
    - Come porciones pequeñas y frecuentes
    - Mantente hidratado
    
    ⚕️ *Si el dolor persiste >24h, consulta un médico*
    """
    
    HEADACHE_TEMPLATE = """
    🤕 **Dolor de Cabeza - Recomendaciones**
    
    Para dolor de cabeza leve a moderado:
    
    **💊 Opciones de venta libre:**
    - **Paracetamol** (500-1000mg): Cada 6-8h, máx 4g/día
    - **Ibuprofeno** (400-600mg): Cada 6-8h con alimentos
    - **Aspirina** (500mg): Cada 4-6h, no en menores de 16 años
    
    **🚨 Consulta médico urgente si tienes:**
    - Dolor súbito e intenso ("el peor dolor de mi vida")
    - Dolor con fiebre alta y rigidez de cuello
    - Dolor con pérdida de visión o habla
    - Dolor tras golpe en la cabeza
    
    **💡 Consejos adicionales:**
    - Mantente hidratado
    - Descansa en lugar oscuro y silencioso
    - Aplica compresas frías o calientes
    
    ⚕️ *Si es frecuente o severo, consulta un médico*
    """
    
    FEVER_TEMPLATE = """
    🌡️ **Fiebre - Recomendaciones**
    
    Para reducir la fiebre en adultos:
    
    **💊 Opciones de venta libre:**
    - **Paracetamol** (500-1000mg): Cada 6-8h
    - **Ibuprofeno** (400-600mg): Cada 6-8h con alimentos
    - **Aspirina** (500mg): Solo en adultos
    
    **🚨 Consulta médico urgente si tienes:**
    - Fiebre >39.5°C persistente
    - Dificultad para respirar
    - Dolor de pecho severo
    - Confusión o delirio
    - Erupción cutánea
    
    **💡 Medidas adicionales:**
    - Bebe abundantes líquidos
    - Usa ropa ligera
    - Baños de agua tibia
    - Reposo
    
    ⚕️ *Fiebre >3 días requiere evaluación médica*
    """
    
    # Templates para medicamentos específicos
    PARACETAMOL_INFO = """
    💊 **Paracetamol (Acetaminofén)**
    
    **Indicaciones**: Dolor leve/moderado, fiebre
    **Dosis adultos**: 500-1000mg cada 6-8h
    **Dosis máxima**: 4g/día (4000mg)
    **Administración**: Con o sin alimentos
    
    ⚠️ **Precauciones**:
    - No exceder dosis máxima
    - Cuidado con otros medicamentos que contengan paracetamol
    - Precaución en problemas hepáticos
    - Evitar alcohol durante tratamiento
    
    **Efectos secundarios**: Raros en dosis normales
    
    ⚕️ *Consulta médico si síntomas persisten >3 días*
    """
    
    IBUPROFEN_INFO = """
    💊 **Ibuprofeno**
    
    **Indicaciones**: Dolor, inflamación, fiebre
    **Dosis adultos**: 400-600mg cada 6-8h
    **Dosis máxima**: 2400mg/día
    **Administración**: Con alimentos
    
    ⚠️ **Contraindicaciones**:
    - Úlcera gástrica activa
    - Problemas renales graves
    - Alergia a AINEs
    - Último trimestre embarazo
    
    **Efectos secundarios**: Molestias gástricas, mareos
    
    ⚕️ *No usar >10 días sin supervisión médica*
    """
    
    OMEPRAZOLE_INFO = """
    💊 **Omeprazol**
    
    **Indicaciones**: Acidez, gastritis, reflujo
    **Dosis adultos**: 20mg una vez al día
    **Administración**: En ayunas, 30-60 min antes del desayuno
    **Duración**: Máximo 14 días sin supervisión médica
    
    ⚠️ **Precauciones**:
    - No usar si alergia a inhibidores de bomba de protones
    - Puede reducir absorción de vitamina B12 y magnesio
    - Interactúa con algunos medicamentos
    
    **Efectos secundarios**: Dolor de cabeza, náuseas, diarrea
    
    ⚕️ *Si síntomas persisten >14 días, consulta médico*
    """
    
    # Template por defecto
    DEFAULT_MEDICATION_TEMPLATE = """
    💊 **Consulta sobre Medicamentos**
    
    Para brindarte información específica, necesito saber:
    - ¿Qué medicamento te interesa?
    - ¿Qué aspecto específico? (dosis, efectos, interacciones)
    
    Puedo ayudarte con información sobre:
    - Dosis recomendadas
    - Efectos secundarios
    - Interacciones
    - Contraindicaciones
    - Forma de administración
    
    ⚕️ *Siempre consulta con un profesional de la salud*
    """
    
    UNKNOWN_MEDICATION_TEMPLATE = """
    💊 **Información sobre {medication}**
    
    Para información específica sobre {medication}, te recomiendo:
    
    1. **Consultar el prospecto** del medicamento
    2. **Hablar con tu farmacéutico** de confianza
    3. **Consultar con tu médico** si tienes dudas
    
    **Información general que puedo proporcionar**:
    - Dosis habituales
    - Precauciones generales
    - Interacciones conocidas
    
    ¿Hay algún aspecto específico sobre {medication} que te interese?
    
    ⚕️ *Siempre sigue las indicaciones de tu médico*
    """
    
    @classmethod
    def get_symptom_response(cls, symptoms: List[str]) -> str:
        """Obtiene respuesta basada en síntomas detectados"""
        symptoms_lower = [s.lower() for s in symptoms]
        
        if any(s in symptoms_lower for s in ["dolor de estomago", "dolor de estómago", "duele el estomago", "duele el estómago", "dolor estomacal", "acidez", "gastritis"]):
            return cls.STOMACH_PAIN_TEMPLATE
        elif any(s in symptoms_lower for s in ["dolor de cabeza", "cefalea", "migraña", "jaqueca"]):
            return cls.HEADACHE_TEMPLATE
        elif any(s in symptoms_lower for s in ["fiebre", "temperatura", "calentura"]):
            return cls.FEVER_TEMPLATE
        else:
            return cls.DEFAULT_MEDICATION_TEMPLATE
    
    @classmethod
    def get_medication_info(cls, medication: str) -> str:
        """Obtiene información específica de un medicamento"""
        med_lower = medication.lower()
        
        if med_lower in ["paracetamol", "acetaminofen", "acetaminofén"]:
            return cls.PARACETAMOL_INFO
        elif med_lower in ["ibuprofeno", "ibuprofen"]:
            return cls.IBUPROFEN_INFO
        elif med_lower in ["omeprazol", "omeprazole"]:
            return cls.OMEPRAZOLE_INFO
        else:
            return cls.UNKNOWN_MEDICATION_TEMPLATE.format(medication=medication.title())
    
    @classmethod
    def get_openai_prompt(cls, message: str, medications: List[str], symptoms: List[str]) -> Dict[str, str]:
        """Genera prompt para OpenAI basado en el contexto"""
        
        context = f"""
        Mensaje del usuario: {message}
        Medicamentos mencionados: {', '.join(medications) if medications else 'Ninguno'}
        Síntomas detectados: {', '.join(symptoms) if symptoms else 'Ninguno'}
        """
        
        user_prompt = f"""
        {context}
        
        Proporciona información farmacéutica apropiada siguiendo estas pautas:
        1. Si hay síntomas específicos, recomienda medicamentos de venta libre apropiados
        2. Si se menciona un medicamento específico, proporciona información detallada
        3. Incluye siempre precauciones y cuándo consultar un médico
        4. Usa formato claro con emojis y secciones organizadas
        5. Termina con disclaimer de seguridad apropiado
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
