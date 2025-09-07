"""
Modelos para el sistema de conversaciones de Farmacia IA
"""

import uuid
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from security.encryption import encrypt_field, decrypt_field


class Conversation(models.Model):
    """
    Modelo para conversaciones con usuarios
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.CharField(max_length=255, help_text="ID del usuario (puede ser anónimo)")
    ip_address = models.GenericIPAddressField(help_text="Dirección IP del usuario")
    started_at = models.DateTimeField(default=timezone.now)
    closed_at = models.DateTimeField(null=True, blank=True)
    meta = models.JSONField(default=dict, blank=True, help_text="Metadatos adicionales")
    encrypted = models.BooleanField(default=False, help_text="Si el contenido está encriptado")
    
    # Campos para tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['user_id']),
            models.Index(fields=['ip_address']),
            models.Index(fields=['started_at']),
            models.Index(fields=['closed_at']),
        ]
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user_id}"
    
    @property
    def is_active(self):
        """Retorna True si la conversación está activa"""
        return self.closed_at is None
    
    @property
    def duration(self):
        """Retorna la duración de la conversación"""
        if self.closed_at:
            return self.closed_at - self.started_at
        return timezone.now() - self.started_at
    
    def close(self):
        """Cierra la conversación"""
        if not self.closed_at:
            self.closed_at = timezone.now()
            self.save(update_fields=['closed_at'])
    
    def get_message_count(self):
        """Retorna el número de mensajes en la conversación"""
        return self.messages.count()


class Message(models.Model):
    """
    Modelo para mensajes dentro de conversaciones
    """
    ROLE_CHOICES = [
        ('user', 'Usuario'),
        ('assistant', 'Asistente'),
        ('system', 'Sistema'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, 
        on_delete=models.CASCADE, 
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField(help_text="Contenido del mensaje (puede estar encriptado)")
    tokens = models.PositiveIntegerField(default=0, help_text="Número de tokens utilizados")
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Campos para encriptación
    _encrypted_content = models.TextField(null=True, blank=True, help_text="Contenido encriptado")
    
    class Meta:
        db_table = 'messages'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['role']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Message {self.id} - {self.role} in {self.conversation.id}"
    
    def save(self, *args, **kwargs):
        """Override save para manejar encriptación automática"""
        if self.conversation.encrypted and self.content and not self._encrypted_content:
            # Encriptar contenido si la conversación requiere encriptación
            self._encrypted_content = encrypt_field(self.content)
            # Limpiar contenido en texto plano por seguridad
            self.content = "[ENCRYPTED]"
        super().save(*args, **kwargs)
    
    def get_content(self):
        """Retorna el contenido desencriptado si es necesario"""
        if self.conversation.encrypted and self._encrypted_content:
            return decrypt_field(self._encrypted_content)
        return self.content
    
    def set_content(self, content):
        """Establece el contenido, encriptando si es necesario"""
        if self.conversation.encrypted:
            self._encrypted_content = encrypt_field(content)
            self.content = "[ENCRYPTED]"
        else:
            self.content = content
            self._encrypted_content = None


class AuditLog(models.Model):
    """
    Modelo para auditoría de eventos del sistema
    """
    EVENT_CHOICES = [
        ('conversation_started', 'Conversación Iniciada'),
        ('conversation_closed', 'Conversación Cerrada'),
        ('message_sent', 'Mensaje Enviado'),
        ('message_received', 'Mensaje Recibido'),
        ('user_login', 'Usuario Logueado'),
        ('user_logout', 'Usuario Deslogueado'),
        ('api_call', 'Llamada API'),
        ('error_occurred', 'Error Ocurrido'),
        ('security_event', 'Evento de Seguridad'),
        ('rate_limit_exceeded', 'Límite de Velocidad Excedido'),
    ]
    
    id = models.AutoField(primary_key=True)
    event = models.CharField(max_length=50, choices=EVENT_CHOICES)
    actor = models.CharField(
        max_length=255, 
        help_text="Hash del user_id o identificador del actor"
    )
    conversation = models.ForeignKey(
        Conversation, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='audit_logs'
    )
    payload = models.JSONField(
        default=dict, 
        blank=True, 
        help_text="Datos adicionales del evento (sin PII)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.CharField(
        max_length=50, 
        help_text="IP enmascarada para auditoría"
    )
    
    class Meta:
        db_table = 'audit_logs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event']),
            models.Index(fields=['actor']),
            models.Index(fields=['created_at']),
            models.Index(fields=['conversation']),
        ]
    
    def __str__(self):
        return f"AuditLog {self.id} - {self.event} by {self.actor}"
    
    @classmethod
    def log_event(cls, event, actor, conversation=None, payload=None, ip_address=None):
        """
        Método de conveniencia para crear logs de auditoría
        """
        from security.encryption import hash_user_id, mask_ip_address
        
        # Hash del actor para no almacenar PII
        hashed_actor = hash_user_id(str(actor)) if actor else "anonymous"
        
        # Enmascarar IP
        masked_ip = mask_ip_address(ip_address) if ip_address else "unknown"
        
        return cls.objects.create(
            event=event,
            actor=hashed_actor,
            conversation=conversation,
            payload=payload or {},
            ip_address=masked_ip
        )


# Señales para auditoría automática
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver(post_save, sender=Conversation)
def audit_conversation_created(sender, instance, created, **kwargs):
    """Auditar creación de conversaciones"""
    if created:
        AuditLog.log_event(
            event='conversation_started',
            actor=instance.user_id,
            conversation=instance,
            payload={
                'conversation_id': str(instance.id),
                'encrypted': instance.encrypted,
            },
            ip_address=instance.ip_address
        )

@receiver(post_save, sender=Message)
def audit_message_created(sender, instance, created, **kwargs):
    """Auditar creación de mensajes"""
    if created:
        AuditLog.log_event(
            event='message_sent' if instance.role == 'user' else 'message_received',
            actor=instance.conversation.user_id,
            conversation=instance.conversation,
            payload={
                'message_id': str(instance.id),
                'role': instance.role,
                'tokens': instance.tokens,
                'content_length': len(instance.get_content()),
            },
            ip_address=instance.conversation.ip_address
        )