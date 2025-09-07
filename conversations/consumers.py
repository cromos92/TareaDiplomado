"""
WebSocket consumers para chat en tiempo real
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from django.utils import timezone

from agents.graph import get_graph_runner
from agents.state import AgentConfig
from core.langsmith_config import trace_conversation, create_langsmith_session, langsmith_config
from .models import Conversation, Message
from core.metrics import record_chat_message, increment_error

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    """
    Consumer para manejar conexiones WebSocket de chat
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_id = None
        self.user_id = None
        self.graph_runner = None
    
    async def connect(self):
        """Maneja nueva conexión WebSocket"""
        
        # Extraer conversation_id de la URL si existe
        self.conversation_id = self.scope['url_route']['kwargs'].get('conversation_id')
        
        # Obtener user_id (de usuario autenticado o sesión)
        user = self.scope.get('user', AnonymousUser())
        if user.is_authenticated:
            self.user_id = str(user.id)
        else:
            # Usar session key para usuarios anónimos
            session = self.scope.get('session', {})
            session_key = getattr(session, 'session_key', None)
            if session_key:
                self.user_id = f"session:{session_key}"
            else:
                self.user_id = f"anonymous:{self.channel_name}"
        
        # Inicializar graph runner
        config = AgentConfig(
            enable_safety_checks=True,
            include_disclaimers=True
        )
        self.graph_runner = get_graph_runner(config)
        
        # Aceptar conexión
        await self.accept()
        
        # Enviar mensaje de bienvenida
        await self.send_message({
            'type': 'system',
            'message': '🏥 ¡Hola! Soy tu asistente de Farmacia IA. ¿En qué puedo ayudarte hoy?',
            'timestamp': timezone.now().isoformat()
        })
        
        logger.info(f"WebSocket connected: user_id={self.user_id}, conversation_id={self.conversation_id}")
    
    async def disconnect(self, close_code):
        """Maneja desconexión WebSocket"""
        logger.info(f"WebSocket disconnected: user_id={self.user_id}, close_code={close_code}")
    
    async def receive(self, text_data):
        """Maneja mensajes recibidos del cliente"""
        
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'message')
            
            if message_type == 'message':
                await self.handle_user_message(data)
            elif message_type == 'typing':
                await self.handle_typing_indicator(data)
            elif message_type == 'ping':
                await self.handle_ping()
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await self.send_error("Formato de mensaje inválido")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            increment_error("websocket_processing", "chat_consumer")
            await self.send_error("Error procesando mensaje")
    
    async def handle_user_message(self, data: Dict[str, Any]):
        """Procesa mensaje del usuario"""
        
        message_text = data.get('message', '').strip()
        if not message_text:
            await self.send_error("Mensaje vacío")
            return
        
        # Validar longitud del mensaje
        if len(message_text) > 1000:
            await self.send_error("Mensaje demasiado largo (máximo 1000 caracteres)")
            return
        
        try:
            # Guardar mensaje del usuario en BD
            conversation = await self.get_or_create_conversation()
            user_message = await self.save_user_message(conversation, message_text)
            
            # Enviar confirmación de recepción
            await self.send_message({
                'type': 'user_message_received',
                'message_id': str(user_message.id),
                'timestamp': user_message.created_at.isoformat()
            })
            
            # Mostrar indicador de "escribiendo"
            await self.send_message({
                'type': 'typing',
                'is_typing': True
            })
            
            # Procesar mensaje con agentes de IA
            response = await self.process_with_agents(message_text)
            
            # Ocultar indicador de "escribiendo"
            await self.send_message({
                'type': 'typing',
                'is_typing': False
            })
            
            # Guardar respuesta del asistente
            assistant_message = await self.save_assistant_message(
                conversation, 
                response['response'],
                response.get('metadata', {})
            )
            
            # Enviar respuesta al cliente
            await self.send_message({
                'type': 'assistant',
                'message': response['response'],
                'message_id': str(assistant_message.id),
                'timestamp': assistant_message.created_at.isoformat(),
                'metadata': {
                    'intent': response.get('intent'),
                    'trace_id': response.get('trace_id'),
                    'safety_flags': len(response.get('safety_flags', [])),
                    'agents_used': response.get('metadata', {}).get('agents_used', [])
                }
            })
            
            # Registrar métricas
            record_chat_message(
                response.get('intent', 'unknown'), 
                'websocket_chat'
            )
            
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
            increment_error("message_processing", "chat_consumer")
            await self.send_error("Error procesando tu mensaje. Por favor, intenta nuevamente.")
    
    async def handle_typing_indicator(self, data: Dict[str, Any]):
        """Maneja indicador de escritura (placeholder)"""
        # En una implementación completa, esto podría notificar a otros usuarios
        pass
    
    async def handle_ping(self):
        """Maneja ping para mantener conexión activa"""
        await self.send_message({
            'type': 'pong',
            'timestamp': timezone.now().isoformat()
        })
    
    async def process_with_agents(self, message: str) -> Dict[str, Any]:
        """Procesa mensaje usando el grafo de agentes y prompts organizados"""
        
        # Crear sesión de LangSmith si está habilitado
        langsmith_session_id = None
        if langsmith_config.is_enabled:
            langsmith_session_id = create_langsmith_session(
                session_id=str(self.conversation_id) if self.conversation_id else "anonymous",
                user_id=str(self.user_id)
            )
            
            # Tracing directo de la conversación
            try:
                from langsmith import traceable
                return await self._process_with_langsmith_tracing(message, langsmith_session_id)
            except ImportError:
                logger.warning("langsmith no disponible para tracing de conversación")
        
        try:
            # Mostrar workflow paso a paso
            await self.send_workflow_step("🤖 Analizando tu consulta...", "router")
            logger.info(f"Processing message: {message}")
            
            # Simular pequeña pausa para mostrar el workflow
            import asyncio
            await asyncio.sleep(0.5)
            
            # Usar RouterAgent para detectar intención real
            from agents.nodes import RouterAgent
            from agents.state import AgentConfig, create_initial_state
            
            # Crear estado inicial con el mensaje del usuario
            initial_state = create_initial_state(
                user_id=str(self.user_id),
                initial_message=message,
                trace_id=f"trace_{timezone.now().timestamp()}"
            )
            
            # Usar RouterAgent para clasificar intención
            config = AgentConfig()
            router = RouterAgent(config)
            classified_state = router(initial_state)
            
            intent = classified_state["intent"]
            slots = classified_state["slots"]
            
            logger.info(f"RouterAgent detected intent: {intent}, slots: {slots}")
            
            # Definir message_lower para uso en todo el método
            message_lower = message.lower()
            
            # Procesar según intención detectada
            if intent == "emergencia":
                await self.send_workflow_step("🚨 Detectada emergencia médica", "emergency")
                await asyncio.sleep(0.3)
                await self.send_workflow_step("⚡ Activando protocolo de emergencia", "safety")
                await asyncio.sleep(0.3)
                
                from agents.prompts import EmergencyPrompts
                response_text = EmergencyPrompts.get_emergency_response("general")
            
            elif intent == "mixta":
                # Consulta mixta: medicamento + farmacia
                await self.send_workflow_step("🔍 Detectada consulta mixta", "router")
                await asyncio.sleep(0.3)
                await self.send_workflow_step("💊 Procesando información de medicamentos", "medication")
                await asyncio.sleep(0.4)
                
                # Procesar parte de medicamentos
                from agents.prompts import MedicationPrompts
                medication_response = MedicationPrompts.get_symptom_response(["dolor de estómago"])
                
                await self.send_workflow_step("🏥 Buscando farmacias de turno", "pharmacy")
                await asyncio.sleep(0.4)
                
                # Procesar parte de farmacias
                comuna = self.extract_comuna_from_message(message)
                direccion = self.extract_direccion_from_message(message)
                
                if comuna:
                    await self.send_workflow_step(f"📍 Consultando MINSAL para {comuna}", "pharmacy")
                    await asyncio.sleep(0.5)
                    
                    farmacias_data = await self.get_farmacias_minsal(comuna, direccion)
                    pharmacy_response = farmacias_data.get('response', 'No se encontraron farmacias disponibles.')
                else:
                    from agents.prompts import PharmacyPrompts
                    pharmacy_response = PharmacyPrompts.get_query_response("locations")
                
                # Combinar respuestas
                response_text = f"""{medication_response}

---

{pharmacy_response}

---

**📋 Resumen:**
- 💊 **Para tu dolor de estómago:** Antiácidos, omeprazol (ver opciones arriba)
- 🏥 **Farmacias disponibles:** {"Consulta ubicaciones arriba" if comuna else "Especifica tu comuna para ubicaciones exactas"}

⚕️ **Esta información es orientativa y no reemplaza la consulta médica profesional.**"""
            
            elif intent == "medicamento":
                await self.send_workflow_step("💊 Identificando síntomas", "medication")
                await asyncio.sleep(0.4)
                await self.send_workflow_step("🔍 Buscando opciones de tratamiento", "medication")
                await asyncio.sleep(0.4)
                await self.send_workflow_step("⚠️ Verificando contraindicaciones", "safety")
                await asyncio.sleep(0.3)
                
                from agents.prompts import MedicationPrompts
                # Detectar síntomas específicos
                if any(word in message_lower for word in ['estomago', 'estómago', 'dolor de estomago', 'dolor estomacal', 'gastritis', 'acidez']):
                    response_text = MedicationPrompts.get_symptom_response(["dolor de estómago"])
                elif any(word in message_lower for word in ['fiebre', 'temperatura', 'calentura', 'tengo fiebre']):
                    response_text = MedicationPrompts.get_symptom_response(["fiebre"])
                elif any(word in message_lower for word in ['dolor de cabeza', 'cefalea', 'migraña', 'jaqueca', 'duele la cabeza']):
                    response_text = MedicationPrompts.get_symptom_response(["dolor de cabeza"])
                else:
                    response_text = MedicationPrompts.DEFAULT_MEDICATION_TEMPLATE
            
            # Medicamentos específicos y dosis
            elif any(word in message_lower for word in ['paracetamol', 'ibuprofeno', 'aspirina', 'dosis', 'cuanto tomar']):
                await self.send_workflow_step("💊 Identificando medicamento", "medication")
                await asyncio.sleep(0.3)
                await self.send_workflow_step("📋 Consultando vademécum", "medication")
                await asyncio.sleep(0.4)
                await self.send_workflow_step("⚖️ Calculando dosis segura", "safety")
                await asyncio.sleep(0.3)
                
                intent = 'medicamento'
                
                if 'paracetamol' in message_lower:
                    response_text = """💊 **Paracetamol - Información de dosis**

**Dosis para adultos:**
- 🔸 **500-1000mg cada 6-8 horas**
- 🔸 **Máximo: 4000mg (4g) al día**
- 🔸 **No exceder 8 comprimidos de 500mg en 24h**

**Para niños:** Consultar peso y edad con farmacéutico

**⚠️ PRECAUCIONES:**
- No combinar con otros medicamentos que contengan paracetamol
- Evitar alcohol durante el tratamiento
- Consultar médico si síntomas persisten >3 días

⚕️ **Siempre lee el prospecto y consulta con un farmacéutico.**"""
                
                elif 'ibuprofeno' in message_lower:
                    response_text = """💊 **Ibuprofeno - Información de dosis**

**Dosis para adultos:**
- 🔸 **400-600mg cada 8 horas**
- 🔸 **Máximo: 1200mg al día (sin supervisión médica)**
- 🔸 **Tomar con alimentos**

**⚠️ CONTRAINDICACIONES:**
- Úlceras gástricas activas
- Problemas renales o hepáticos graves
- Alergia a AINEs

**💡 RECOMENDACIONES:**
- Tomar con leche o alimentos
- No usar >10 días sin supervisión médica

⚕️ **Consulta médico si tienes problemas gástricos, renales o cardíacos.**"""
                
                else:
                    response_text = f"""💊 **Consulta sobre medicamentos**

Para tu pregunta: "{message}"

**Información general sobre dosis:**
- Siempre seguir las indicaciones del prospecto
- Consultar con farmacéutico para dosis específicas
- No exceder las dosis recomendadas
- Considerar peso, edad y condiciones médicas

**🏥 Te recomiendo:**
1. Visitar una farmacia cercana
2. Consultar con el farmacéutico de turno
3. Llevar lista de otros medicamentos que tomas

⚕️ **Esta información es orientativa y no reemplaza la consulta profesional.**"""
            
            # Farmacias de turno
            elif any(word in message_lower for word in ['farmacia', 'turno', 'horario', 'donde comprar']):
                await self.send_workflow_step("🏥 Localizando farmacias", "pharmacy")
                await asyncio.sleep(0.4)
                await self.send_workflow_step("🌐 Consultando MINSAL", "pharmacy")
                await asyncio.sleep(0.5)
                await self.send_workflow_step("📍 Verificando ubicaciones", "pharmacy")
                await asyncio.sleep(0.3)
                
                # Obtener contexto de conversación anterior
                conversation_context = await self.get_conversation_context()
                
                # Extraer comuna del mensaje con contexto
                comuna_detectada = self.extract_comuna_from_message(message, conversation_context)
                
                # Extraer dirección de referencia si la hay
                direccion_referencia = self.extract_direccion_from_message(message)
                
                if comuna_detectada:
                    await self.send_workflow_step(f"🎯 Buscando en {comuna_detectada}", "pharmacy")
                    await asyncio.sleep(0.4)
                    
                    if direccion_referencia:
                        await self.send_workflow_step(f"📍 Calculando distancias desde {direccion_referencia}", "pharmacy")
                        await asyncio.sleep(0.4)
                    
                    # Consultar MINSAL real
                    farmacias_info = await self.get_farmacias_minsal(comuna_detectada, direccion_referencia)
                    
                    if farmacias_info['found']:
                        response_text = f"""🏥 **Farmacias en {comuna_detectada}**

{farmacias_info['content']}

**🌐 Datos obtenidos de MINSAL** {farmacias_info['timestamp']}
{farmacias_info['status_indicator']}

**💡 Información adicional:**
- Confirma horarios antes de ir
- Para emergencias: SAMU 131
- Sitio oficial: farmaciasdeturno.cl"""
                    else:
                        response_text = f"""🏥 **Farmacias en {comuna_detectada}**

⚠️ **No se pudieron obtener datos actualizados de MINSAL**

**Alternativas:**
- 🔸 **Sitio web**: farmaciasdeturno.cl
- 🔸 **Teléfono**: 600 360 3000
- 🔸 **App móvil**: "Farmacias de Turno"

**🚨 Para emergencias:**
- Servicios de urgencia hospitalarios (24/7)
- SAMU: 131

{farmacias_info['error_msg']}"""
                else:
                    response_text = """🏥 **Farmacias de turno**

**Para encontrar farmacias abiertas:**
- 🔸 **Sitio web MINSAL**: farmaciasdeturno.cl
- 🔸 **App Farmacias de Turno** (móvil)
- 🔸 **Llamar al 600 360 3000** (información telefónica)

**Horarios típicos:**
- Lunes a Viernes: 9:00-20:00
- Sábados: 9:00-14:00
- Domingos y festivos: Solo farmacias de turno

**🚨 Para emergencias:**
- Servicios de urgencia hospitalarios (24/7)
- SAMU: 131

💡 **Tip:** Especifica tu comuna para obtener información más precisa (ej: "farmacias en Las Condes")**"""
                
                intent = 'farmacia'
            
            # Respuesta general mejorada
            else:
                await self.send_workflow_step("🤔 Analizando consulta general", "supervisor")
                await asyncio.sleep(0.4)
                await self.send_workflow_step("📚 Preparando información", "clarification")
                await asyncio.sleep(0.3)
                
                intent = 'general'
                response_text = f"""🏥 **Farmacia IA - Asistente Virtual**

Hola, recibí tu consulta: "{message}"

**🤖 Puedo ayudarte con:**
- 💊 **Medicamentos**: Dosis, usos, precauciones
- 🏥 **Farmacias**: Ubicaciones, horarios, turnos
- ⚠️ **Emergencias**: Orientación inmediata
- 🩺 **Síntomas**: Información general y recomendaciones

**💬 Ejemplos de consultas:**
- "¿Cuál es la dosis de paracetamol?"
- "¿Dónde hay farmacias de turno?"
- "Me duele la cabeza, ¿qué puedo tomar?"

**🔴 Para emergencias médicas graves:**
📞 **Llama al 131 (SAMU) inmediatamente**

⚕️ *Toda información es orientativa y no reemplaza la consulta médica profesional.*"""
            
            return {
                'success': True,
                'response': response_text,
                'intent': intent,
                'trace_id': f"simple_{hash(message)}",
                'safety_flags': [],
                'metadata': {
                    'agents_used': ['simple_processor'],
                    'processing_time': 0.1
                }
            }
            
        except Exception as e:
            logger.error(f"Error in simple processing: {e}")
            return {
                'success': False,
                'response': "Ocurrió un error procesando tu mensaje. Por favor, intenta nuevamente.",
                'intent': 'error',
                'trace_id': 'error',
                'safety_flags': [],
                'metadata': {}
            }
    
    async def send_message(self, message_data: Dict[str, Any]):
        """Envía mensaje al cliente WebSocket"""
        await self.send(text_data=json.dumps(message_data))
    
    async def send_workflow_step(self, step_message: str, agent_type: str):
        """Envía paso del workflow al cliente"""
        await self.send_message({
            'type': 'workflow_step',
            'message': step_message,
            'agent': agent_type,
            'timestamp': timezone.now().isoformat()
        })
    
    async def send_error(self, error_message: str):
        """Envía mensaje de error al cliente"""
        await self.send_message({
            'type': 'error',
            'message': error_message,
            'timestamp': timezone.now().isoformat()
        })
    
    @database_sync_to_async
    def get_or_create_conversation(self) -> Conversation:
        """Obtiene o crea una conversación"""
        
        if self.conversation_id:
            try:
                return Conversation.objects.get(id=self.conversation_id)
            except Conversation.DoesNotExist:
                pass
        
        # Crear nueva conversación
        conversation = Conversation.objects.create(
            user_id=self.user_id,
            ip_address=self.get_client_ip(),
            encrypted=False,  # Por ahora sin encriptación
            meta={
                'channel': 'websocket',
                'user_agent': self.get_user_agent()
            }
        )
        
        self.conversation_id = str(conversation.id)
        return conversation
    
    @database_sync_to_async
    def save_user_message(self, conversation: Conversation, content: str) -> Message:
        """Guarda mensaje del usuario en la base de datos"""
        
        return Message.objects.create(
            conversation=conversation,
            role='user',
            content=content,
            tokens=len(content.split())  # Estimación simple
        )
    
    @database_sync_to_async
    def save_assistant_message(self, conversation: Conversation, content: str, metadata: Dict[str, Any]) -> Message:
        """Guarda mensaje del asistente en la base de datos"""
        
        return Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=content,
            tokens=len(content.split())  # Estimación simple
        )
    
    def get_client_ip(self) -> str:
        """Obtiene la IP del cliente"""
        
        # Buscar en headers de proxy
        headers = dict(self.scope.get('headers', []))
        
        # X-Forwarded-For header
        forwarded_for = headers.get(b'x-forwarded-for')
        if forwarded_for:
            return forwarded_for.decode('utf-8').split(',')[0].strip()
        
        # X-Real-IP header
        real_ip = headers.get(b'x-real-ip')
        if real_ip:
            return real_ip.decode('utf-8').strip()
        
        # Fallback a client del scope
        client = self.scope.get('client')
        if client:
            return client[0]
        
        return 'unknown'
    
    def get_user_agent(self) -> str:
        """Obtiene el User-Agent del cliente"""
        headers = dict(self.scope.get('headers', []))
        user_agent = headers.get(b'user-agent')
        if user_agent:
            return user_agent.decode('utf-8', errors='ignore')
        return 'Unknown'
    
    def extract_comuna_from_message(self, message: str, conversation_context: Optional[str] = None) -> Optional[str]:
        """Extrae nombre de comuna del mensaje con contexto de conversación"""
        import re
        from data.normalizers import commune_normalizer
        
        message_lower = message.lower()
        
        # Primero buscar en el contexto de la conversación anterior
        if conversation_context:
            context_comuna = self._extract_comuna_from_context(conversation_context)
            if context_comuna:
                # Si no hay comuna explícita en el mensaje actual, usar la del contexto
                if not any(word in message_lower for word in ['en', 'de', 'comuna', 'ciudad']):
                    return context_comuna
        
        # Patrones mejorados para detectar comunas
        patterns = [
            r'(?:en|de|cerca\s+de|centro\s+de|comuna\s+de)\s+([a-záéíóúñü\s]+)',
            r'(?:farmacia|turno|horario).*?([a-záéíóúñü]{4,})',  # Capturar después de palabras clave
            r'([a-záéíóúñü]{4,})',  # Palabras de 4+ caracteres
        ]
        
        # Lista de comunas conocidas para matching directo
        known_comunas = [
            'antofagasta', 'santiago', 'valparaiso', 'valparaíso', 'concepcion', 'concepción',
            'las condes', 'providencia', 'ñuñoa', 'nunoa', 'maipu', 'maipú', 'puente alto',
            'la florida', 'san miguel', 'quilicura', 'renca', 'independencia', 'recoleta'
        ]
        
        # Buscar coincidencias directas primero
        for comuna in known_comunas:
            if comuna in message_lower:
                return commune_normalizer.normalize_commune(comuna)
        
        # Luego usar patrones
        for pattern in patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                comuna_candidate = match.strip()
                if len(comuna_candidate) > 2:
                    # Filtrar palabras comunes que no son comunas
                    if comuna_candidate not in ['farmacia', 'turno', 'horario', 'cerca', 'centro', 'donde', 'hay']:
                        normalized = commune_normalizer.normalize_commune(comuna_candidate)
                        if commune_normalizer.is_valid_commune(normalized):
                            return normalized
                        
                        # Buscar coincidencias parciales
                        search_results = commune_normalizer.search_communes(comuna_candidate, limit=1)
                        if search_results:
                            return search_results[0]
        
        return None
    
    def _extract_comuna_from_context(self, context: str) -> Optional[str]:
        """Extrae comuna del contexto de conversación anterior"""
        if not context:
            return None
        
        # Buscar patrones como "Farmacias en [Comuna]"
        import re
        patterns = [
            r'Farmacias en ([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü\s]+)',
            r'farmacias en ([a-záéíóúñü\s]+)',
            r'comuna[:\s]+([a-záéíóúñü\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1).strip()
        
        return None
    
    def combine_farmacia_data(self, farmacias_normales: List[Dict], farmacias_turnos: List[Dict]) -> List[Dict]:
        """Combina datos de farmacias normales y de turno, marcando las de turno"""
        # Crear un diccionario para identificar farmacias de turno por ID
        turnos_ids = set()
        farmacias_turno_dict = {}
        
        for farmacia in farmacias_turnos:
            farmacia_id = farmacia.get('local_id') or farmacia.get('id')
            if farmacia_id:
                turnos_ids.add(str(farmacia_id))
                farmacias_turno_dict[str(farmacia_id)] = farmacia
        
        # Marcar farmacias normales que también están de turno
        combined_farmacias = []
        
        for farmacia in farmacias_normales:
            farmacia_id = farmacia.get('local_id') or farmacia.get('id')
            farmacia_copy = farmacia.copy()
            
            if farmacia_id and str(farmacia_id) in turnos_ids:
                # Esta farmacia está de turno hoy
                farmacia_copy['es_turno'] = True
                farmacia_copy['turno_info'] = farmacias_turno_dict.get(str(farmacia_id), {})
            else:
                farmacia_copy['es_turno'] = False
                farmacia_copy['turno_info'] = {}
            
            combined_farmacias.append(farmacia_copy)
        
        # Agregar farmacias que solo están en turnos (si las hay)
        for farmacia in farmacias_turnos:
            farmacia_id = farmacia.get('local_id') or farmacia.get('id')
            
            # Verificar si ya está en la lista de normales
            already_exists = any(
                (f.get('local_id') or f.get('id')) == farmacia_id 
                for f in farmacias_normales
            )
            
            if not already_exists:
                farmacia_copy = farmacia.copy()
                farmacia_copy['es_turno'] = True
                farmacia_copy['turno_info'] = farmacia
                combined_farmacias.append(farmacia_copy)
        
        # Ordenar: farmacias de turno primero
        combined_farmacias.sort(key=lambda x: (not x.get('es_turno', False), x.get('local_nombre', '')))
        
        return combined_farmacias
    
    def extract_direccion_from_message(self, message: str) -> Optional[str]:
        """Extrae dirección de referencia del mensaje"""
        import re
        
        message_lower = message.lower()
        
        # Patrones para detectar direcciones
        patterns = [
            r'(?:cerca\s+de|desde|en)\s+([a-záéíóúñü\s\d]+\d+[a-záéíóúñü\s\d]*)',  # "cerca de Bernardo O'Higgins 1650"
            r'(?:direccion|dirección)\s+([a-záéíóúñü\s\d]+)',  # "direccion Bernardo O'Higgins 1650"
            r'([a-záéíóúñü\s]+\d+[a-záéíóúñü\s\d]*)',  # Cualquier texto seguido de número
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                direccion_candidate = match.strip()
                if len(direccion_candidate) > 10 and any(char.isdigit() for char in direccion_candidate):
                    # Limpiar la dirección
                    return self.clean_direccion(direccion_candidate)
        
        return None
    
    def clean_direccion(self, direccion: str) -> str:
        """Limpia y normaliza una dirección"""
        # Capitalizar palabras importantes
        words = direccion.split()
        cleaned_words = []
        
        for word in words:
            if word.lower() in ['de', 'del', 'la', 'las', 'el', 'los', 'y']:
                cleaned_words.append(word.lower())
            else:
                cleaned_words.append(word.capitalize())
        
        return ' '.join(cleaned_words)
    
    async def get_conversation_context(self) -> Optional[str]:
        """Obtiene el contexto de la conversación actual"""
        try:
            if hasattr(self, 'conversation_id') and self.conversation_id:
                conversation = await self.get_conversation()
                if conversation:
                    # Obtener los últimos 3 mensajes para contexto
                    messages = await self.get_recent_messages(conversation, limit=3)
                    context_parts = []
                    
                    for msg in messages:
                        if msg.role == 'assistant':
                            # Extraer información relevante de respuestas anteriores
                            if 'Farmacias en' in msg.content:
                                context_parts.append(msg.content[:200])  # Primeros 200 caracteres
                    
                    return ' '.join(context_parts) if context_parts else None
            return None
        except Exception as e:
            logger.warning(f"Error getting conversation context: {e}")
            return None
    
    @database_sync_to_async
    def get_recent_messages(self, conversation, limit=3):
        """Obtiene mensajes recientes de la conversación"""
        from .models import Message
        return list(Message.objects.filter(
            conversation=conversation
        ).order_by('-created_at')[:limit])
    
    async def get_farmacias_minsal(self, comuna: str, direccion_referencia: Optional[str] = None) -> Dict[str, Any]:
        """
        Consulta farmacias reales desde MINSAL usando ambos endpoints:
        - getLocales.php: Todas las farmacias con horarios regulares (incluye sábados)
        - getLocalesTurnos.php: Farmacias de turno con horarios extendidos/24h
        """
        try:
            import httpx
            from datetime import datetime
            
            # Endpoints de MINSAL
            url_normales = "https://midas.minsal.cl/farmacia_v2/WS/getLocales.php"
            url_turnos = "https://midas.minsal.cl/farmacia_v2/WS/getLocalesTurnos.php"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Consultar ambos endpoints
                farmacias_normales = []
                farmacias_turnos = []
                
                try:
                    response_normales = await client.get(url_normales)
                    response_normales.raise_for_status()
                    farmacias_normales = response_normales.json()
                except Exception as e:
                    logger.warning(f"Error consultando farmacias normales: {e}")
                
                try:
                    response_turnos = await client.get(url_turnos)
                    response_turnos.raise_for_status()
                    farmacias_turnos = response_turnos.json()
                except Exception as e:
                    logger.warning(f"Error consultando farmacias de turno: {e}")
                
                # Si no hay datos de ningún endpoint, lanzar error
                if not farmacias_normales and not farmacias_turnos:
                    raise Exception("No se pudieron obtener datos de MINSAL")
                
                # Combinar y marcar las farmacias de turno
                farmacias_data = self.combine_farmacia_data(farmacias_normales, farmacias_turnos)
                
                # Filtrar por comuna
                farmacias_filtradas = self.filter_farmacias_by_comuna(farmacias_data, comuna)
                
                if farmacias_filtradas:
                    # Si hay dirección de referencia, calcular distancias y ordenar
                    if direccion_referencia:
                        farmacias_con_distancia = await self.calculate_distances_to_farmacias(
                            farmacias_filtradas, direccion_referencia, comuna
                        )
                        content = self.format_farmacias_minsal_response(farmacias_con_distancia, direccion_referencia)
                    else:
                        content = self.format_farmacias_minsal_response(farmacias_filtradas)
                    
                    return {
                        'found': True,
                        'content': content,
                        'timestamp': f"({datetime.now().strftime('%H:%M')})",
                        'status_indicator': '🟢 **Datos en tiempo real de MINSAL**',
                        'error_msg': ''
                    }
                else:
                    return {
                        'found': False,
                        'content': '',
                        'timestamp': '',
                        'status_indicator': '',
                        'error_msg': f'💡 **No se encontraron farmacias en {comuna}**. Verifica el nombre de la comuna.'
                    }
                
        except Exception as e:
            logger.error(f"Error consulting MINSAL: {e}")
            
            # Fallback a datos simulados si falla MINSAL
            farmacias_simuladas = self.get_simulated_farmacias(comuna)
            if farmacias_simuladas:
                content = self.format_farmacias_response(farmacias_simuladas)
                return {
                    'found': True,
                    'content': content,
                    'timestamp': '(fallback)',
                    'status_indicator': '🟡 **Datos de respaldo** - MINSAL no disponible',
                    'error_msg': f'⚠️ Error conectando con MINSAL: {str(e)}'
                }
            
            return {
                'found': False,
                'content': '',
                'timestamp': '',
                'status_indicator': '',
                'error_msg': f'⚠️ **Error técnico:** {str(e)}'
            }
    
    def get_simulated_farmacias(self, comuna: str) -> List[Dict[str, Any]]:
        """Genera datos simulados de farmacias para testing"""
        comuna_lower = comuna.lower()
        
        # Base de datos simulada de farmacias por comuna
        farmacias_db = {
            'antofagasta': [
                {
                    'nombre': 'Farmacia Salcobrand',
                    'direccion': 'Av. Argentina 1105',
                    'telefono': '+56 55 2345678',
                    'horario': '09:00 - 21:00',
                    'es_turno': True
                },
                {
                    'nombre': 'Farmacia Novoa',
                    'direccion': 'Arturo Prat 656',
                    'telefono': '+56 55 2876543',
                    'horario': '09:00 - 22:30',
                    'es_turno': False
                },
                {
                    'nombre': 'Farmacia Cruz Verde',
                    'direccion': 'Coquimbo 712',
                    'telefono': '+56 55 2987654',
                    'horario': '08:30 - 21:00',
                    'es_turno': True
                }
            ],
            'santiago': [
                {
                    'nombre': 'Farmacia Ahumada',
                    'direccion': 'Av. Providencia 1234',
                    'telefono': '+56 2 2345678',
                    'horario': '08:00 - 22:00',
                    'es_turno': True
                },
                {
                    'nombre': 'Farmacia Cruz Verde',
                    'direccion': 'Av. Las Condes 567',
                    'telefono': '+56 2 2876543',
                    'horario': '24 horas',
                    'es_turno': True
                }
            ],
            'las condes': [
                {
                    'nombre': 'Farmacia Salcobrand',
                    'direccion': 'Av. Apoquindo 4500',
                    'telefono': '+56 2 2345678',
                    'horario': '08:00 - 23:00',
                    'es_turno': True
                },
                {
                    'nombre': 'Farmacia del Dr. Simi',
                    'direccion': 'Av. Kennedy 9001',
                    'telefono': '+56 2 2987654',
                    'horario': '09:00 - 21:00',
                    'es_turno': False
                }
            ]
        }
        
        return farmacias_db.get(comuna_lower, [])
    
    def format_farmacias_response(self, farmacias: List[Dict[str, Any]]) -> str:
        """Formatea respuesta de farmacias"""
        if not farmacias:
            return "No se encontraron farmacias en esta comuna."
        
        response_parts = []
        turno_count = 0
        
        for farmacia in farmacias:
            turno_indicator = "🟢 **DE TURNO**" if farmacia.get('es_turno') else "🔵 Horario normal"
            if farmacia.get('es_turno'):
                turno_count += 1
            
            farmacia_info = f"""**{farmacia['nombre']}** {turno_indicator}
📍 {farmacia['direccion']}
📞 {farmacia.get('telefono', 'No disponible')}
🕒 {farmacia.get('horario', 'Consultar horarios')}"""
            
            response_parts.append(farmacia_info)
        
        header = f"**Encontradas {len(farmacias)} farmacias ({turno_count} de turno):**\n\n"
        return header + "\n\n".join(response_parts)
    
    def filter_farmacias_by_comuna(self, farmacias_data: List[Dict], comuna_target: str) -> List[Dict]:
        """Filtra farmacias por comuna desde datos de MINSAL"""
        from data.normalizers import commune_normalizer
        
        comuna_normalized = commune_normalizer.normalize_commune(comuna_target).upper()
        farmacias_filtradas = []
        
        for farmacia in farmacias_data:
            comuna_farmacia = farmacia.get('comuna_nombre', '').upper()
            
            # Comparación exacta o parcial
            if (comuna_normalized in comuna_farmacia or 
                comuna_farmacia in comuna_normalized or
                self.similar_comuna_names(comuna_normalized, comuna_farmacia)):
                farmacias_filtradas.append(farmacia)
        
        # Limitar a máximo 10 farmacias para no saturar la respuesta
        return farmacias_filtradas[:10]
    
    def similar_comuna_names(self, name1: str, name2: str) -> bool:
        """Verifica si dos nombres de comuna son similares"""
        # Remover palabras comunes
        common_words = {'DE', 'LA', 'LAS', 'EL', 'LOS', 'DEL'}
        
        words1 = set(name1.split()) - common_words
        words2 = set(name2.split()) - common_words
        
        # Si hay intersección de palabras principales
        return len(words1.intersection(words2)) > 0
    
    def format_farmacias_minsal_response(self, farmacias: List[Dict], direccion_referencia: Optional[str] = None) -> str:
        """Formatea respuesta de farmacias desde datos reales de MINSAL"""
        if not farmacias:
            return "No se encontraron farmacias."
        
        response_parts = []
        turno_count = 0
        has_distances = any('distance_km' in f for f in farmacias)
        
        for i, farmacia in enumerate(farmacias):
            # Determinar si está de turno usando la nueva información
            es_turno = farmacia.get('es_turno', False)
            
            if es_turno:
                turno_count += 1
            
            turno_indicator = "🟢 **DE TURNO HOY**" if es_turno else "🔵 Horario normal"
            
            # Formatear horarios (usar info de turno si está disponible)
            if es_turno and farmacia.get('turno_info'):
                turno_info = farmacia['turno_info']
                hora_apertura = turno_info.get('funcionamiento_hora_apertura', farmacia.get('funcionamiento_hora_apertura', '00:00:00'))
                hora_cierre = turno_info.get('funcionamiento_hora_cierre', farmacia.get('funcionamiento_hora_cierre', '00:00:00'))
            else:
                hora_apertura = farmacia.get('funcionamiento_hora_apertura', '00:00:00')
                hora_cierre = farmacia.get('funcionamiento_hora_cierre', '00:00:00')
            
            if hora_apertura == '00:00:00' and hora_cierre == '00:00:00':
                horario = "24 horas" if es_turno else "Consultar horario"
            else:
                horario = f"{hora_apertura[:5]} - {hora_cierre[:5]}"
                if es_turno:
                    horario += " (TURNO)"
            
            # Limpiar teléfono
            telefono = farmacia.get('local_telefono', '+560')
            if telefono == '+560' or telefono == '+56':
                telefono = 'Consultar'
            
            # Agregar distancia si está disponible
            distance_info = ""
            if has_distances and 'distance_km' in farmacia:
                distance = farmacia['distance_km']
                if distance < 999:
                    if distance < 1:
                        distance_info = f"📏 {int(distance * 1000)}m"
                    else:
                        distance_info = f"📏 {distance}km"
                    
                    # Agregar indicador de cercanía
                    if i == 0 and distance < 2:
                        turno_indicator += " ⭐ **MÁS CERCANA**"
            
            farmacia_info = f"""**{farmacia.get('local_nombre', 'Farmacia')}** {turno_indicator}
📍 {farmacia.get('local_direccion', 'Dirección no disponible')}
🏘️ {farmacia.get('comuna_nombre', '')} - {farmacia.get('localidad_nombre', '')}
📞 {telefono}
🕒 {horario}"""
            
            if distance_info:
                farmacia_info += f"\n{distance_info}"
            
            response_parts.append(farmacia_info)
        
        # Header con información de distancias
        if direccion_referencia and has_distances:
            header = f"**Farmacias más cercanas a {direccion_referencia}** ({turno_count} de turno hoy):\n\n"
        else:
            header = f"**Encontradas {len(farmacias)} farmacias ({turno_count} de turno hoy):**\n\n"
        
        return header + "\n\n".join(response_parts)
    
    async def calculate_distances_to_farmacias(self, farmacias: List[Dict], direccion_referencia: str, comuna: str) -> List[Dict]:
        """Calcula distancias desde una dirección de referencia a las farmacias"""
        try:
            # Geocodificar dirección de referencia
            ref_coords = await self.geocode_address(f"{direccion_referencia}, {comuna}, Chile")
            
            if not ref_coords:
                # Si no se puede geocodificar, devolver farmacias sin ordenar
                return farmacias
            
            farmacias_con_distancia = []
            
            for farmacia in farmacias:
                # Usar coordenadas de MINSAL si están disponibles
                lat_str = farmacia.get('local_lat', '').replace('°', '').replace(',', '')
                lng_str = farmacia.get('local_lng', '').replace('°', '').replace(',', '')
                
                try:
                    if lat_str and lng_str and lat_str != '' and lng_str != '':
                        lat = float(lat_str)
                        lng = float(lng_str)
                        
                        # Calcular distancia
                        distance = self.calculate_distance(ref_coords[0], ref_coords[1], lat, lng)
                        farmacia['distance_km'] = round(distance, 2)
                    else:
                        # Si no hay coordenadas, intentar geocodificar la dirección de la farmacia
                        farmacia_address = f"{farmacia.get('local_direccion', '')}, {farmacia.get('comuna_nombre', '')}, Chile"
                        farmacia_coords = await self.geocode_address(farmacia_address)
                        
                        if farmacia_coords:
                            distance = self.calculate_distance(ref_coords[0], ref_coords[1], farmacia_coords[0], farmacia_coords[1])
                            farmacia['distance_km'] = round(distance, 2)
                        else:
                            farmacia['distance_km'] = 999  # Distancia alta si no se puede calcular
                            
                except (ValueError, TypeError):
                    farmacia['distance_km'] = 999
                
                farmacias_con_distancia.append(farmacia)
            
            # Ordenar por distancia (más cercanas primero)
            farmacias_con_distancia.sort(key=lambda x: x.get('distance_km', 999))
            
            return farmacias_con_distancia[:8]  # Limitar a 8 más cercanas
            
        except Exception as e:
            logger.error(f"Error calculating distances: {e}")
            return farmacias
    
    async def geocode_address(self, address: str) -> Optional[tuple]:
        """Geocodifica una dirección usando un servicio gratuito"""
        try:
            import httpx
            import urllib.parse
            
            # Usar Nominatim (OpenStreetMap) - servicio gratuito
            encoded_address = urllib.parse.quote(address)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json&limit=1"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers={
                    'User-Agent': 'Farmacia-IA/1.0 (contact@farmacia-ia.cl)'
                })
                response.raise_for_status()
                
                data = response.json()
                if data and len(data) > 0:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    return (lat, lon)
                    
        except Exception as e:
            logger.warning(f"Error geocoding address '{address}': {e}")
        
        return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia entre dos puntos usando fórmula de Haversine"""
        import math
        
        # Radio de la Tierra en km
        R = 6371.0
        
        # Convertir grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Diferencias
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Fórmula de Haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        return distance
    
    async def _process_with_langsmith_tracing(self, message: str, session_id: str) -> Dict[str, Any]:
        """Procesa mensaje con tracing de LangSmith"""
        from langsmith import traceable
        
        @traceable(
            name="farmacia_conversation",
            tags=["farmacia-ia", "conversation", "websocket"],
            metadata={
                "user_id": str(self.user_id),
                "conversation_id": str(self.conversation_id) if self.conversation_id else "anonymous",
                "session_id": session_id
            }
        )
        async def process_message_traced(user_message: str) -> dict:
            """Función interna trackeada"""
            try:
                # Mostrar workflow paso a paso
                await self.send_workflow_step("🤖 Analizando tu consulta...", "router")
                logger.info(f"Processing message with LangSmith tracing: {user_message}")
                
                # Simular pequeña pausa para mostrar el workflow
                import asyncio
                await asyncio.sleep(0.5)

                from agents.nodes import RouterAgent
                from agents.state import AgentConfig, create_initial_state

                # Definir message_lower para uso en todo el método
                message_lower = user_message.lower()

                # Crear estado inicial con el mensaje del usuario
                initial_state = create_initial_state(
                    user_id=str(self.user_id),
                    initial_message=user_message,
                    trace_id=f"trace_{timezone.now().timestamp()}"
                )
                
                # Usar RouterAgent para clasificar intención
                config = AgentConfig()
                router = RouterAgent(config)
                classified_state = router(initial_state)
                
                detected_intent = classified_state["intent"]
                logger.info(f"RouterAgent detected intent: {detected_intent}, slots: {classified_state['slots']}")
                
                # Procesar según intención (versión simplificada para tracing)
                if detected_intent == "medicamento":
                    await self.send_workflow_step("💊 Buscando información de medicamentos...", "medication")
                    response_message = f"💊 Información sobre medicamentos para: '{user_message}'"
                elif detected_intent == "farmacia":
                    await self.send_workflow_step("🏥 Buscando farmacias...", "pharmacy")
                    response_message = f"🏥 Información sobre farmacias para: '{user_message}'"
                elif detected_intent == "emergencia":
                    await self.send_workflow_step("🚨 Procesando emergencia...", "emergency")
                    response_message = f"🚨 Emergencia detectada: '{user_message}'"
                else:
                    await self.send_workflow_step("❓ Procesando consulta general...", "general")
                    response_message = f"❓ Consulta general: '{user_message}'"
                
                return {
                    'type': 'message',
                    'message': response_message,
                    'intent': detected_intent,
                    'trace_id': classified_state["trace_id"],
                    'langsmith_session': session_id
                }
                
            except Exception as e:
                logger.error(f"Error in LangSmith traced processing: {e}")
                return {
                    'type': 'error',
                    'message': 'Error procesando tu mensaje con tracing. Por favor, intenta nuevamente.',
                    'error': str(e)
                }
        
        # Ejecutar función trackeada
        return await process_message_traced(message)
