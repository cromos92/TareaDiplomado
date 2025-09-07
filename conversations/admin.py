"""
Configuración del admin para la app conversations
"""

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Conversation, Message, AuditLog


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """Admin para el modelo Conversation"""
    
    list_display = [
        'id', 'user_id', 'ip_address', 'started_at', 
        'closed_at', 'is_active', 'message_count', 'encrypted'
    ]
    list_filter = [
        'encrypted', 'started_at', 'closed_at', 'created_at'
    ]
    search_fields = ['user_id', 'ip_address', 'id']
    readonly_fields = ['id', 'created_at', 'updated_at', 'duration_display']
    date_hierarchy = 'started_at'
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('id', 'user_id', 'ip_address')
        }),
        ('Timestamps', {
            'fields': ('started_at', 'closed_at', 'duration_display')
        }),
        ('Configuración', {
            'fields': ('encrypted', 'meta')
        }),
        ('Auditoría', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def is_active(self, obj):
        """Muestra si la conversación está activa"""
        if obj.is_active:
            return format_html(
                '<span style="color: green;">✓ Activa</span>'
            )
        return format_html(
            '<span style="color: red;">✗ Cerrada</span>'
        )
    is_active.short_description = 'Estado'
    
    def message_count(self, obj):
        """Muestra el número de mensajes"""
        count = obj.get_message_count()
        url = reverse('admin:conversations_message_changelist')
        return format_html(
            '<a href="{}?conversation__id__exact={}">{} mensajes</a>',
            url, obj.id, count
        )
    message_count.short_description = 'Mensajes'
    
    def duration_display(self, obj):
        """Muestra la duración de la conversación"""
        duration = obj.duration
        if duration:
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "N/A"
    duration_display.short_description = 'Duración'
    
    actions = ['close_conversations']
    
    def close_conversations(self, request, queryset):
        """Acción para cerrar conversaciones seleccionadas"""
        count = 0
        for conversation in queryset:
            if conversation.is_active:
                conversation.close()
                count += 1
        
        self.message_user(
            request, 
            f'{count} conversaciones cerradas exitosamente.'
        )
    close_conversations.short_description = "Cerrar conversaciones seleccionadas"


class MessageInline(admin.TabularInline):
    """Inline para mostrar mensajes en la conversación"""
    model = Message
    extra = 0
    readonly_fields = ['id', 'created_at', 'content_preview']
    fields = ['role', 'content_preview', 'tokens', 'created_at']
    
    def content_preview(self, obj):
        """Muestra una preview del contenido del mensaje"""
        content = obj.get_content()
        if len(content) > 100:
            return content[:100] + "..."
        return content
    content_preview.short_description = 'Contenido'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin para el modelo Message"""
    
    list_display = [
        'id', 'conversation_link', 'role', 'content_preview', 
        'tokens', 'created_at'
    ]
    list_filter = ['role', 'created_at', 'conversation__encrypted']
    search_fields = ['conversation__id', 'conversation__user_id', 'content']
    readonly_fields = ['id', 'created_at', 'content_display']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Información Básica', {
            'fields': ('id', 'conversation', 'role')
        }),
        ('Contenido', {
            'fields': ('content_display', 'tokens')
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )
    
    def conversation_link(self, obj):
        """Link a la conversación"""
        url = reverse('admin:conversations_conversation_change', args=[obj.conversation.id])
        return format_html(
            '<a href="{}">{}</a>',
            url, str(obj.conversation.id)[:8] + "..."
        )
    conversation_link.short_description = 'Conversación'
    
    def content_preview(self, obj):
        """Preview del contenido"""
        content = obj.get_content()
        if len(content) > 50:
            return content[:50] + "..."
        return content
    content_preview.short_description = 'Contenido'
    
    def content_display(self, obj):
        """Muestra el contenido completo (solo lectura)"""
        content = obj.get_content()
        return format_html('<pre>{}</pre>', content)
    content_display.short_description = 'Contenido Completo'


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    """Admin para el modelo AuditLog"""
    
    list_display = [
        'id', 'event', 'actor_display', 'conversation_link', 
        'ip_address', 'created_at'
    ]
    list_filter = ['event', 'created_at']
    search_fields = ['actor', 'ip_address', 'event']
    readonly_fields = [
        'id', 'event', 'actor', 'conversation', 
        'payload_display', 'created_at', 'ip_address'
    ]
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Información del Evento', {
            'fields': ('id', 'event', 'actor', 'ip_address')
        }),
        ('Contexto', {
            'fields': ('conversation', 'payload_display')
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        }),
    )
    
    def has_add_permission(self, request):
        """No permitir agregar logs manualmente"""
        return False
    
    def has_change_permission(self, request, obj=None):
        """No permitir editar logs"""
        return False
    
    def has_delete_permission(self, request, obj=None):
        """No permitir eliminar logs (solo superuser)"""
        return request.user.is_superuser
    
    def actor_display(self, obj):
        """Muestra el actor truncado"""
        return obj.actor[:16] + "..." if len(obj.actor) > 16 else obj.actor
    actor_display.short_description = 'Actor'
    
    def conversation_link(self, obj):
        """Link a la conversación si existe"""
        if obj.conversation:
            url = reverse('admin:conversations_conversation_change', args=[obj.conversation.id])
            return format_html(
                '<a href="{}">{}</a>',
                url, str(obj.conversation.id)[:8] + "..."
            )
        return "N/A"
    conversation_link.short_description = 'Conversación'
    
    def payload_display(self, obj):
        """Muestra el payload formateado"""
        import json
        try:
            formatted = json.dumps(obj.payload, indent=2, ensure_ascii=False)
            return format_html('<pre>{}</pre>', formatted)
        except:
            return str(obj.payload)
    payload_display.short_description = 'Payload'


# Personalizar el admin site
admin.site.site_header = "Farmacia IA - Administración"
admin.site.site_title = "Farmacia IA Admin"
admin.site.index_title = "Panel de Administración"