"""
Cliente OpenAI para agentes de Farmacia IA
"""

import os
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from django.conf import settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Cliente para interactuar con la API de OpenAI
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            logger.warning("OpenAI API key not configured. Using placeholder responses.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate_pharmacy_response(
        self, 
        message: str, 
        slots: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Genera respuesta para consultas de farmacia usando OpenAI
        
        Args:
            message: Mensaje del usuario
            slots: Slots extraídos
            context: Contexto adicional
        
        Returns:
            Respuesta generada
        """
        
        if not self.client:
            return self._placeholder_pharmacy_response(message, slots)
        
        try:
            system_prompt = """
            Eres un asistente especializado en consultas farmacéuticas. 
            Proporciona información precisa, segura y útil sobre:
            - Horarios y ubicaciones de farmacias
            - Disponibilidad de medicamentos
            - Consultas generales sobre servicios farmacéuticos
            
            IMPORTANTE:
            - Siempre incluye disclaimers apropiados
            - No proporciones diagnósticos médicos
            - Recomienda consultar con profesionales cuando sea necesario
            - Mantén un tono profesional pero amigable
            """
            
            user_prompt = f"""
            Consulta del usuario: {message}
            
            Información extraída: {slots}
            
            Contexto adicional: {context or {}}
            
            Por favor, proporciona una respuesta útil y segura.
            """
            
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.openai_temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI pharmacy response: {e}")
            return self._placeholder_pharmacy_response(message, slots)
    
    async def generate_medication_response(
        self, 
        message: str, 
        medications: List[str],
        slots: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Genera respuesta para consultas sobre medicamentos usando OpenAI
        
        Args:
            message: Mensaje del usuario
            medications: Lista de medicamentos mencionados
            slots: Slots extraídos
            context: Contexto adicional
        
        Returns:
            Respuesta generada
        """
        
        if not self.client:
            return self._placeholder_medication_response(message, medications)
        
        try:
            system_prompt = """
            Eres un asistente especializado en información farmacológica.
            Proporciona información precisa sobre medicamentos incluyendo:
            - Dosis recomendadas
            - Efectos secundarios
            - Contraindicaciones
            - Interacciones medicamentosas
            - Forma de administración
            
            CRÍTICO:
            - NUNCA proporciones diagnósticos médicos
            - SIEMPRE incluye disclaimers de seguridad
            - Recomienda consultar con médico/farmacéutico
            - Si detectas riesgo, recomienda atención médica inmediata
            - Usa formato claro con emojis apropiados
            """
            
            user_prompt = f"""
            Consulta: {message}
            Medicamentos mencionados: {medications}
            Información extraída: {slots}
            Contexto: {context or {}}
            
            Proporciona información segura y útil sobre estos medicamentos.
            """
            
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.openai_temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI medication response: {e}")
            return self._placeholder_medication_response(message, medications)
    
    async def classify_intent(self, message: str) -> Dict[str, Any]:
        """
        Clasifica la intención del mensaje usando OpenAI
        
        Args:
            message: Mensaje del usuario
        
        Returns:
            Clasificación de intención con confianza
        """
        
        if not self.client:
            return {"intent": "desconocida", "confidence": 0.5}
        
        try:
            system_prompt = """
            Clasifica la intención del usuario en una de estas categorías:
            - farmacia: Consultas generales sobre farmacias, horarios, ubicaciones
            - medicamento: Consultas específicas sobre medicamentos, dosis, efectos
            - emergencia: Situaciones urgentes que requieren atención médica
            - mixta: Consultas que combinan múltiples intenciones
            - desconocida: Intención no clara
            
            Responde SOLO con un JSON: {"intent": "categoria", "confidence": 0.0-1.0}
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Usar modelo más rápido para clasificación
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error classifying intent with OpenAI: {e}")
            return {"intent": "desconocida", "confidence": 0.5}
    
    def _placeholder_pharmacy_response(self, message: str, slots: Dict[str, Any]) -> str:
        """Respuesta placeholder para consultas de farmacia"""
        return """
        🏥 **Consulta de Farmacia**
        
        Gracias por tu consulta sobre servicios farmacéuticos.
        
        **Información general**:
        - Horarios habituales: Lunes a Viernes 9:00-20:00
        - Farmacias de turno: Consulta en tu municipio
        - Servicios disponibles: Medicamentos, consultas, vacunación
        
        Para información específica de tu zona, te recomiendo:
        - Contactar farmacias locales
        - Consultar apps oficiales de tu región
        - Llamar al servicio de información farmacéutica
        
        ⚕️ *Esta información es orientativa. Para consultas específicas, contacta directamente con la farmacia.*
        """
    
    def _placeholder_medication_response(self, message: str, medications: List[str]) -> str:
        """Respuesta placeholder para consultas de medicamentos"""
        meds_text = ", ".join(medications) if medications else "el medicamento consultado"
        
        return f"""
        💊 **Información sobre {meds_text}**
        
        Para obtener información precisa sobre este medicamento, te recomiendo:
        
        **Fuentes confiables**:
        - Prospecto del medicamento
        - Consulta con tu farmacéutico
        - Revisión con tu médico tratante
        
        **Información que puedes solicitar**:
        - Dosis recomendada
        - Efectos secundarios
        - Contraindicaciones
        - Interacciones medicamentosas
        
        ⚠️ **IMPORTANTE**: Nunca modifiques dosis o suspendas medicamentos sin supervisión médica.
        
        ⚕️ *Esta información es orientativa y no reemplaza la consulta médica profesional.*
        """
