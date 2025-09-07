"""
Prompts y templates para el PharmacyAgent
"""

from typing import Dict, List, Any


class PharmacyPrompts:
    """Prompts específicos para consultas de farmacia"""
    
    # Sistema de prompts para OpenAI
    SYSTEM_PROMPT = """
    Eres un asistente especializado en información de farmacias. Tu función es:
    
    1. Proporcionar información sobre ubicaciones y horarios de farmacias
    2. Ayudar con consultas sobre disponibilidad de medicamentos
    3. Informar sobre precios y alternativas genéricas
    4. Conectar con servicios de farmacia de turno
    
    IMPORTANTE:
    - Usa datos reales cuando estén disponibles (MINSAL API)
    - Proporciona información práctica y útil
    - Incluye horarios y contactos cuando sea posible
    - Sugiere alternativas cuando no hay disponibilidad
    
    Responde de forma clara y orientada a la acción.
    """
    
    # Templates para consultas específicas
    PHARMACY_HOURS_TEMPLATE = """
    🕒 **Horarios de Farmacia**
    
    **Horarios habituales**:
    - Lunes a Viernes: 9:00 - 20:00
    - Sábados: 9:00 - 14:00
    - Domingos: Farmacias de turno
    
    **Farmacias de turno**: Consulta en tu municipio o app local
    **Emergencias**: Servicios hospitalarios 24/7
    
    ℹ️ *Los horarios pueden variar según la ubicación*
    """
    
    PHARMACY_PRICES_TEMPLATE = """
    💰 **Información de Precios**
    
    Los precios de medicamentos varían según:
    - Marca vs. genérico
    - Farmacia
    - Descuentos disponibles
    - Cobertura de seguro
    
    **Recomendaciones**:
    - Consulta genéricos (mismo principio activo)
    - Compara precios entre farmacias
    - Pregunta por descuentos disponibles
    
    ℹ️ *Para precios específicos, consulta directamente en farmacia*
    """
    
    GENERIC_MEDICATIONS_TEMPLATE = """
    💊 **Medicamentos Genéricos**
    
    **¿Qué son?**: Medicamentos con el mismo principio activo que el original
    **Ventajas**:
    - Mismo efecto terapéutico
    - Precio más económico
    - Misma calidad y seguridad
    
    **Equivalencia**: Aprobada por autoridades sanitarias
    **Disponibilidad**: Pregunta en farmacia por alternativas genéricas
    
    ✅ *Los genéricos son una opción segura y económica*
    """
    
    PHARMACY_LOCATIONS_TEMPLATE = """
    📍 **Ubicaciones de Farmacias**
    
    Para encontrar farmacias cerca de ti:
    
    **🔍 Opciones de búsqueda:**
    - Farmacias de turno por comuna
    - Farmacias 24 horas
    - Farmacias con servicios especiales
    
    **📱 Recursos útiles:**
    - App oficial de tu municipio
    - Sitio web MINSAL
    - Llamar al 131 para emergencias
    
    **🏥 Servicios adicionales:**
    - Inyecciones y vacunas
    - Control de presión arterial
    - Entrega a domicilio
    
    ℹ️ *Especifica tu comuna para información más precisa*
    """
    
    PRESCRIPTION_INFO_TEMPLATE = """
    📋 **Información sobre Recetas**
    
    **Medicamentos con receta médica:**
    - Requieren prescripción válida
    - No se pueden vender sin receta
    - Vigencia limitada (generalmente 30 días)
    
    **Medicamentos de venta libre:**
    - Disponibles sin receta
    - Asesoría farmacéutica recomendada
    - Límites de cantidad por compra
    
    **📝 Consejos para recetas:**
    - Verifica fecha de vigencia
    - Lleva documento de identidad
    - Consulta sobre genéricos disponibles
    
    ℹ️ *El farmacéutico puede orientarte sobre alternativas*
    """
    
    # Template por defecto
    DEFAULT_PHARMACY_TEMPLATE = """
    🏥 **Asistente de Farmacia**
    
    Puedo ayudarte con:
    - Información sobre medicamentos
    - Horarios y ubicaciones de farmacias
    - Consultas sobre genéricos vs. marca
    - Precios y disponibilidad
    - Consejos generales de salud
    
    ¿En qué aspecto específico necesitas ayuda?
    
    ⚕️ *Recuerda que esta información es orientativa*
    """
    
    # Template para farmacias de turno
    DUTY_PHARMACY_TEMPLATE = """
    🏥 **Farmacias de Turno - {location}**
    
    {pharmacy_list}
    
    **📞 Información adicional:**
    - Llama antes de ir para confirmar disponibilidad
    - Algunas pueden tener horarios especiales
    - Servicios de emergencia disponibles 24/7
    
    **🚨 En caso de emergencia:**
    - Llama al 131 (SAMU)
    - Acude al servicio de urgencias más cercano
    
    🌐 *Datos obtenidos de MINSAL - {timestamp}*
    """
    
    NO_PHARMACIES_FOUND_TEMPLATE = """
    ❌ **No se encontraron farmacias**
    
    No se encontraron farmacias de turno en {location}.
    
    **🔍 Alternativas:**
    - Verifica la ortografía de la comuna
    - Intenta con una comuna cercana
    - Consulta farmacias 24 horas en hospitales
    
    **📞 Contactos útiles:**
    - SAMU: 131
    - Información MINSAL: 600 360 7777
    
    **🏥 Servicios de emergencia:**
    - Hospital más cercano
    - SAPU (Servicio de Atención Primaria de Urgencia)
    
    ℹ️ *Intenta especificar una comuna válida de Chile*
    """
    
    @classmethod
    def get_query_response(cls, query_type: str, **kwargs) -> str:
        """Obtiene respuesta basada en el tipo de consulta"""
        
        if query_type == "hours":
            return cls.PHARMACY_HOURS_TEMPLATE
        elif query_type == "prices":
            return cls.PHARMACY_PRICES_TEMPLATE
        elif query_type == "generic":
            return cls.GENERIC_MEDICATIONS_TEMPLATE
        elif query_type == "locations":
            return cls.PHARMACY_LOCATIONS_TEMPLATE
        elif query_type == "prescription":
            return cls.PRESCRIPTION_INFO_TEMPLATE
        elif query_type == "duty_pharmacies":
            return cls.DUTY_PHARMACY_TEMPLATE.format(**kwargs)
        elif query_type == "no_pharmacies":
            return cls.NO_PHARMACIES_FOUND_TEMPLATE.format(**kwargs)
        else:
            return cls.DEFAULT_PHARMACY_TEMPLATE
    
    @classmethod
    def format_pharmacy_list(cls, pharmacies: List[Dict], reference_address: str = None) -> str:
        """Formatea lista de farmacias para mostrar"""
        if not pharmacies:
            return "No se encontraron farmacias disponibles."
        
        formatted_list = []
        turno_count = 0
        
        for i, pharmacy in enumerate(pharmacies):
            # Determinar si está de turno
            es_turno = pharmacy.get('es_turno', False)
            if es_turno:
                turno_count += 1
            
            turno_indicator = "🟢 **DE TURNO HOY**" if es_turno else "🔵 Horario normal"
            
            # Formatear horarios
            hora_apertura = pharmacy.get('funcionamiento_hora_apertura', '00:00:00')
            hora_cierre = pharmacy.get('funcionamiento_hora_cierre', '00:00:00')
            
            if hora_apertura == '00:00:00' and hora_cierre == '00:00:00':
                horario = "24 horas" if es_turno else "Consultar horario"
            else:
                horario = f"{hora_apertura[:5]} - {hora_cierre[:5]}"
                if es_turno:
                    horario += " (TURNO)"
            
            # Información de distancia
            distance_info = ""
            if 'distance_km' in pharmacy and pharmacy['distance_km'] < 999:
                distance = pharmacy['distance_km']
                if distance < 1:
                    distance_info = f"\n📏 {int(distance * 1000)}m"
                else:
                    distance_info = f"\n📏 {distance}km"
                
                if i == 0 and distance < 2:
                    turno_indicator += " ⭐ **MÁS CERCANA**"
            
            # Teléfono
            telefono = pharmacy.get('local_telefono', 'Consultar')
            if telefono in ['+560', '+56', '']:
                telefono = 'Consultar'
            
            pharmacy_info = f"""**{pharmacy.get('local_nombre', 'Farmacia')}** {turno_indicator}
📍 {pharmacy.get('local_direccion', 'Dirección no disponible')}
🏘️ {pharmacy.get('comuna_nombre', '')} - {pharmacy.get('localidad_nombre', '')}
📞 {telefono}
🕒 {horario}{distance_info}"""
            
            formatted_list.append(pharmacy_info)
        
        header = f"**Encontradas {len(pharmacies)} farmacias ({turno_count} de turno hoy):**\n\n"
        if reference_address:
            header = f"**Farmacias más cercanas a {reference_address}** ({turno_count} de turno hoy):\n\n"
        
        return header + "\n\n".join(formatted_list)
    
    @classmethod
    def get_openai_prompt(cls, message: str, location: str = None, query_type: str = None) -> Dict[str, str]:
        """Genera prompt para OpenAI basado en el contexto"""
        
        context = f"""
        Mensaje del usuario: {message}
        Ubicación mencionada: {location or 'No especificada'}
        Tipo de consulta: {query_type or 'General'}
        """
        
        user_prompt = f"""
        {context}
        
        Proporciona información útil sobre farmacias siguiendo estas pautas:
        1. Si se solicita ubicación específica, enfócate en esa área
        2. Si se pregunta por horarios, proporciona información general y específica
        3. Si se consulta sobre precios, explica factores que los afectan
        4. Si se buscan farmacias de turno, explica cómo encontrarlas
        5. Usa formato claro con emojis y secciones organizadas
        6. Incluye información práctica y contactos útiles
        """
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_prompt
        }
