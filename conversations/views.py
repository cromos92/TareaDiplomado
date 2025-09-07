"""
Vistas para la app conversations
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from .models import Conversation, Message


def conversation_list(request):
    """Lista todas las conversaciones del usuario"""
    user_id = request.user.id if request.user.is_authenticated else request.session.session_key
    
    conversations = Conversation.objects.filter(
        user_id=str(user_id)
    ).order_by('-started_at')[:20]
    
    context = {
        'conversations': conversations,
        'user_id': user_id,
    }
    
    return render(request, 'conversations/list.html', context)


def conversation_detail(request, conversation_id):
    """Detalle de una conversación específica"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    
    # Verificar permisos (usuario propietario o staff)
    user_id = str(request.user.id) if request.user.is_authenticated else request.session.session_key
    
    if conversation.user_id != user_id and not request.user.is_staff:
        messages.error(request, 'No tienes permisos para ver esta conversación.')
        return redirect('conversations:conversation-list')
    
    messages = conversation.messages.all().order_by('created_at')
    
    context = {
        'conversation': conversation,
        'messages': messages,
    }
    
    return render(request, 'conversations/detail.html', context)


@require_http_methods(["GET", "POST"])
def new_conversation(request):
    """Crea una nueva conversación"""
    if request.method == 'POST':
        user_id = str(request.user.id) if request.user.is_authenticated else request.session.session_key
        
        # Obtener IP del cliente
        ip_address = request.META.get('HTTP_X_FORWARDED_FOR')
        if ip_address:
            ip_address = ip_address.split(',')[0].strip()
        else:
            ip_address = request.META.get('REMOTE_ADDR', '127.0.0.1')
        
        # Crear nueva conversación
        conversation = Conversation.objects.create(
            user_id=user_id,
            ip_address=ip_address,
            encrypted=request.POST.get('encrypted', False)
        )
        
        messages.success(request, 'Nueva conversación creada exitosamente.')
        return redirect('conversations:conversation-detail', conversation_id=conversation.id)
    
    return render(request, 'conversations/new.html')


def chat_view(request, conversation_id=None):
    """Vista principal del chat"""
    
    conversation = None
    if conversation_id:
        try:
            conversation = get_object_or_404(Conversation, id=conversation_id)
            # Verificar que el usuario tenga acceso a esta conversación
            if request.user.is_authenticated:
                if conversation.user_id != str(request.user.id):
                    conversation = None
        except:
            # Si el ID no es válido, crear nueva conversación
            conversation = None

    # Obtener conversaciones recientes del usuario si está autenticado
    recent_conversations = []
    if request.user.is_authenticated:
        recent_conversations = Conversation.objects.filter(
            user_id=str(request.user.id)
        ).order_by('-started_at')[:5]

    context = {
        'title': 'Chat - Farmacia IA',
        'conversation': conversation,
        'conversation_id': conversation_id,
        'recent_conversations': recent_conversations,
        'user_authenticated': request.user.is_authenticated,
    }
    
    return render(request, 'chat/chat.html', context)


@require_http_methods(["GET"])
def conversation_messages(request, conversation_id):
    """API endpoint para obtener mensajes de una conversación"""
    
    try:
        conversation = get_object_or_404(Conversation, id=conversation_id)
        
        # Verificar permisos
        user_id = str(request.user.id) if request.user.is_authenticated else request.session.session_key
        if conversation.user_id != user_id and not request.user.is_staff:
            return JsonResponse({'error': 'No tienes permisos para ver esta conversación'}, status=403)
        
        # Obtener mensajes
        messages = conversation.messages.all().order_by('created_at')
        
        messages_data = []
        for message in messages:
            messages_data.append({
                'id': str(message.id),
                'role': message.role,
                'content': message.content,
                'created_at': message.created_at.isoformat(),
                'tokens': message.tokens
            })
        
        return JsonResponse({
            'conversation_id': str(conversation.id),
            'messages': messages_data,
            'total_messages': len(messages_data)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)