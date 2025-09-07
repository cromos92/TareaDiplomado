"""
Serializers para la API de conversations
"""

from rest_framework import serializers
from .models import Conversation, Message, AuditLog


class MessageSerializer(serializers.ModelSerializer):
    """Serializer para mensajes"""
    
    content = serializers.SerializerMethodField()
    
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'role', 'content', 
            'tokens', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_content(self, obj):
        """Retorna el contenido desencriptado si es necesario"""
        return obj.get_content()
    
    def validate_role(self, value):
        """Valida que el rol sea válido"""
        valid_roles = ['user', 'assistant', 'system']
        if value not in valid_roles:
            raise serializers.ValidationError(
                f"Role must be one of: {', '.join(valid_roles)}"
            )
        return value
    
    def validate_content(self, value):
        """Valida el contenido del mensaje"""
        if not value or not value.strip():
            raise serializers.ValidationError("Content cannot be empty")
        
        if len(value) > 10000:  # Límite de 10k caracteres
            raise serializers.ValidationError("Content is too long (max 10000 characters)")
        
        return value.strip()


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer para conversaciones"""
    
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    duration = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user_id', 'ip_address', 'started_at', 
            'closed_at', 'meta', 'encrypted', 'created_at', 
            'updated_at', 'messages', 'message_count', 
            'duration', 'is_active'
        ]
        read_only_fields = [
            'id', 'user_id', 'ip_address', 'started_at', 
            'created_at', 'updated_at'
        ]
    
    def get_message_count(self, obj):
        """Retorna el número de mensajes"""
        return obj.get_message_count()
    
    def get_duration(self, obj):
        """Retorna la duración en segundos"""
        duration = obj.duration
        if duration:
            return duration.total_seconds()
        return None
    
    def get_is_active(self, obj):
        """Retorna si la conversación está activa"""
        return obj.is_active
    
    def validate_meta(self, value):
        """Valida que meta sea un dict válido"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Meta must be a valid JSON object")
        return value


class ConversationListSerializer(serializers.ModelSerializer):
    """Serializer simplificado para listas de conversaciones"""
    
    message_count = serializers.SerializerMethodField()
    last_message_at = serializers.SerializerMethodField()
    is_active = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'started_at', 'closed_at', 'encrypted',
            'message_count', 'last_message_at', 'is_active'
        ]
    
    def get_message_count(self, obj):
        """Retorna el número de mensajes"""
        return obj.get_message_count()
    
    def get_last_message_at(self, obj):
        """Retorna la fecha del último mensaje"""
        last_message = obj.messages.order_by('-created_at').first()
        if last_message:
            return last_message.created_at
        return obj.started_at
    
    def get_is_active(self, obj):
        """Retorna si la conversación está activa"""
        return obj.is_active


class AuditLogSerializer(serializers.ModelSerializer):
    """Serializer para logs de auditoría (solo lectura)"""
    
    class Meta:
        model = AuditLog
        fields = [
            'id', 'event', 'actor', 'conversation', 
            'payload', 'created_at', 'ip_address'
        ]
        read_only_fields = '__all__'


class ChatRequestSerializer(serializers.Serializer):
    """Serializer para requests del chat API"""
    
    message = serializers.CharField(
        max_length=10000,
        help_text="Contenido del mensaje del usuario"
    )
    conversation_id = serializers.UUIDField(
        required=False,
        allow_null=True,
        help_text="ID de conversación existente (opcional)"
    )
    encrypted = serializers.BooleanField(
        default=False,
        help_text="Si la conversación debe ser encriptada"
    )
    
    def validate_message(self, value):
        """Valida el mensaje"""
        if not value or not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer para responses del chat API"""
    
    conversation_id = serializers.UUIDField()
    messages = MessageSerializer(many=True)
    
    class Meta:
        fields = ['conversation_id', 'messages']
