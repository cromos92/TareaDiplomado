"""
API endpoints para conversations usando DRF
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from django.utils import timezone
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
from .models import Conversation, Message, AuditLog
from .serializers import ConversationSerializer, MessageSerializer


# Rate limiter setup
# limiter = Limiter(key_func=get_remote_address)


class ChatAPIView(APIView):
    """
    API endpoint principal para chat con rate limiting
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        """
        Procesa mensajes de chat
        """
        try:
            # Rate limiting manual (slowapi se integra mejor con FastAPI)
            from django.core.cache import cache
            from django.conf import settings
            
            # Obtener IP del cliente
            ip = self._get_client_ip(request)
            
            # Verificar rate limit
            cache_key = f"chat_rate_limit:{ip}"
            current_count = cache.get(cache_key, 0)
            max_requests = getattr(settings, 'RATE_LIMIT_REQUESTS', 100)
            
            if current_count >= max_requests:
                # Log rate limit exceeded
                AuditLog.log_event(
                    event='rate_limit_exceeded',
                    actor=self._get_user_id(request),
                    payload={
                        'endpoint': '/api/chat/',
                        'ip': ip,
                        'count': current_count
                    },
                    ip_address=ip
                )
                
                return Response({
                    'error': 'Rate limit exceeded',
                    'detail': f'Maximum {max_requests} requests per hour'
                }, status=status.HTTP_429_TOO_MANY_REQUESTS)
            
            # Incrementar contador
            cache.set(cache_key, current_count + 1, 3600)  # 1 hora
            
            # Procesar mensaje
            data = request.data
            conversation_id = data.get('conversation_id')
            message_content = data.get('message', '')
            
            if not message_content.strip():
                return Response({
                    'error': 'Message content is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Obtener o crear conversación
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(id=conversation_id)
                except Conversation.DoesNotExist:
                    return Response({
                        'error': 'Conversation not found'
                    }, status=status.HTTP_404_NOT_FOUND)
            else:
                # Crear nueva conversación
                conversation = Conversation.objects.create(
                    user_id=self._get_user_id(request),
                    ip_address=ip,
                    encrypted=data.get('encrypted', False)
                )
            
            # Crear mensaje del usuario
            user_message = Message.objects.create(
                conversation=conversation,
                role='user',
                content=message_content,
                tokens=len(message_content.split())  # Estimación simple
            )
            
            # TODO: Aquí iría la integración con el agente de IA
            # Por ahora, respuesta placeholder
            assistant_response = "Esta es una respuesta placeholder del asistente de farmacia."
            
            # Crear mensaje del asistente
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=assistant_response,
                tokens=len(assistant_response.split())
            )
            
            # Log del evento
            AuditLog.log_event(
                event='api_call',
                actor=self._get_user_id(request),
                conversation=conversation,
                payload={
                    'endpoint': '/api/chat/',
                    'user_message_length': len(message_content),
                    'assistant_message_length': len(assistant_response),
                },
                ip_address=ip
            )
            
            return Response({
                'conversation_id': str(conversation.id),
                'messages': [
                    {
                        'id': str(user_message.id),
                        'role': 'user',
                        'content': user_message.get_content(),
                        'created_at': user_message.created_at.isoformat()
                    },
                    {
                        'id': str(assistant_message.id),
                        'role': 'assistant',
                        'content': assistant_message.get_content(),
                        'created_at': assistant_message.created_at.isoformat()
                    }
                ]
            })
        
        except Exception as e:
            # Log error
            AuditLog.log_event(
                event='error_occurred',
                actor=self._get_user_id(request),
                payload={
                    'endpoint': '/api/chat/',
                    'error': str(e),
                },
                ip_address=self._get_client_ip(request)
            )
            
            return Response({
                'error': 'Internal server error',
                'detail': 'An error occurred processing your request'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_client_ip(self, request):
        """Obtiene la IP del cliente"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
        return ip
    
    def _get_user_id(self, request):
        """Obtiene el ID del usuario"""
        if request.user.is_authenticated:
            return str(request.user.id)
        
        session_key = request.session.session_key
        if session_key:
            return f"session:{session_key}"
        
        return "anonymous"


class ConversationViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gestionar conversaciones
    """
    serializer_class = ConversationSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        """Filtra conversaciones por usuario"""
        user_id = self._get_user_id()
        return Conversation.objects.filter(user_id=user_id).order_by('-started_at')
    
    def perform_create(self, serializer):
        """Crea conversación con datos del request"""
        ip_address = self._get_client_ip()
        user_id = self._get_user_id()
        
        serializer.save(
            user_id=user_id,
            ip_address=ip_address
        )
    
    @action(detail=True, methods=['post'])
    def close(self, request, pk=None):
        """Cierra una conversación"""
        conversation = self.get_object()
        conversation.close()
        
        return Response({
            'message': 'Conversation closed successfully',
            'closed_at': conversation.closed_at.isoformat()
        })
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Obtiene mensajes de una conversación"""
        conversation = self.get_object()
        messages = conversation.messages.all().order_by('created_at')
        
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    def _get_client_ip(self):
        """Obtiene IP del cliente"""
        request = self.request
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
        return ip
    
    def _get_user_id(self):
        """Obtiene ID del usuario"""
        request = self.request
        if request.user.is_authenticated:
            return str(request.user.id)
        
        session_key = request.session.session_key
        if session_key:
            return f"session:{session_key}"
        
        return "anonymous"


class MessageViewSet(viewsets.ModelViewSet):
    """
    ViewSet para gestionar mensajes
    """
    serializer_class = MessageSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        """Filtra mensajes por conversaciones del usuario"""
        user_id = self._get_user_id()
        return Message.objects.filter(
            conversation__user_id=user_id
        ).order_by('-created_at')
    
    def perform_create(self, serializer):
        """Crea mensaje verificando permisos"""
        conversation = serializer.validated_data['conversation']
        user_id = self._get_user_id()
        
        # Verificar que el usuario puede escribir en esta conversación
        if conversation.user_id != user_id:
            raise PermissionError("Cannot write to this conversation")
        
        serializer.save()
    
    def _get_user_id(self):
        """Obtiene ID del usuario"""
        request = self.request
        if request.user.is_authenticated:
            return str(request.user.id)
        
        session_key = request.session.session_key
        if session_key:
            return f"session:{session_key}"
        
        return "anonymous"
