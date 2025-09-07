from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.cache import never_cache
from django.http import JsonResponse
from django.contrib.auth.models import User
import json

from .forms import CustomUserCreationForm, CustomAuthenticationForm, ProfileUpdateForm


@csrf_protect
@never_cache
def login_view(request):
    """Vista de login"""
    if request.user.is_authenticated:
        return redirect('conversations:chat')
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'¡Bienvenido de vuelta, {user.first_name or user.username}!')
                
                # Redirigir a la página solicitada o al chat
                next_url = request.GET.get('next', reverse('conversations:chat'))
                return redirect(next_url)
            else:
                messages.error(request, 'Usuario o contraseña incorrectos.')
        else:
            messages.error(request, 'Por favor, corrige los errores del formulario.')
    else:
        form = CustomAuthenticationForm()
    
    context = {
        'form': form,
        'title': 'Iniciar Sesión - Farmacia IA'
    }
    return render(request, 'auth/login.html', context)


@csrf_protect
def register_view(request):
    """Vista de registro"""
    if request.user.is_authenticated:
        return redirect('conversations:chat')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            
            # Autenticar y hacer login automático
            user = authenticate(username=username, password=form.cleaned_data.get('password1'))
            if user is not None:
                login(request, user)
                messages.success(request, f'¡Cuenta creada exitosamente! Bienvenido, {user.first_name}!')
                return redirect('conversations:chat')
        else:
            messages.error(request, 'Por favor, corrige los errores del formulario.')
    else:
        form = CustomUserCreationForm()
    
    context = {
        'form': form,
        'title': 'Crear Cuenta - Farmacia IA'
    }
    return render(request, 'auth/register.html', context)


@never_cache
def logout_view(request):
    """Vista de logout"""
    if request.user.is_authenticated:
        user_name = request.user.first_name or request.user.username
        logout(request)
        messages.success(request, f'¡Hasta luego, {user_name}!')
    
    return redirect('home')


@login_required
def profile_view(request):
    """Vista de perfil de usuario"""
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Perfil actualizado correctamente.')
            return redirect('profile')
        else:
            messages.error(request, 'Por favor, corrige los errores del formulario.')
    else:
        form = ProfileUpdateForm(instance=request.user)
    
    # Obtener estadísticas del usuario
    from conversations.models import Conversation, Message
    
    user_conversations = Conversation.objects.filter(user_id=str(request.user.id))
    total_conversations = user_conversations.count()
    total_messages = Message.objects.filter(conversation__in=user_conversations).count()
    
    context = {
        'form': form,
        'title': 'Mi Perfil - Farmacia IA',
        'stats': {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
        }
    }
    return render(request, 'auth/profile.html', context)


@login_required
def delete_account_view(request):
    """Vista para eliminar cuenta"""
    if request.method == 'POST':
        # Verificar contraseña antes de eliminar
        password = request.POST.get('password')
        if request.user.check_password(password):
            # Eliminar conversaciones del usuario
            from conversations.models import Conversation
            Conversation.objects.filter(user_id=str(request.user.id)).delete()
            
            # Eliminar usuario
            user_name = request.user.first_name or request.user.username
            request.user.delete()
            
            messages.success(request, f'Cuenta de {user_name} eliminada correctamente.')
            return redirect('home')
        else:
            messages.error(request, 'Contraseña incorrecta.')
    
    context = {
        'title': 'Eliminar Cuenta - Farmacia IA'
    }
    return render(request, 'auth/delete_account.html', context)


@login_required
def export_data_view(request):
    """Vista para exportar datos del usuario"""
    from conversations.models import Conversation, Message
    from django.http import HttpResponse
    import json
    from datetime import datetime
    
    # Obtener todas las conversaciones del usuario
    conversations = Conversation.objects.filter(user_id=str(request.user.id))
    
    export_data = {
        'user_info': {
            'username': request.user.username,
            'email': request.user.email,
            'first_name': request.user.first_name,
            'last_name': request.user.last_name,
            'date_joined': request.user.date_joined.isoformat(),
        },
        'conversations': []
    }
    
    for conversation in conversations:
        messages = Message.objects.filter(conversation=conversation).order_by('created_at')
        
        conversation_data = {
            'id': str(conversation.id),
            'started_at': conversation.started_at.isoformat(),
            'closed_at': conversation.closed_at.isoformat() if conversation.closed_at else None,
            'messages': []
        }
        
        for message in messages:
            message_data = {
                'role': message.role,
                'content': message.content,
                'created_at': message.created_at.isoformat(),
                'tokens': message.tokens
            }
            conversation_data['messages'].append(message_data)
        
        export_data['conversations'].append(conversation_data)
    
    # Crear respuesta JSON
    response = HttpResponse(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        content_type='application/json; charset=utf-8'
    )
    
    filename = f"farmacia_ia_data_{request.user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    return response


def check_username_availability(request):
    """API endpoint para verificar disponibilidad de username"""
    if request.method == 'GET':
        username = request.GET.get('username', '').strip()
        
        if len(username) < 3:
            return JsonResponse({'available': False, 'message': 'El nombre de usuario debe tener al menos 3 caracteres'})
        
        if User.objects.filter(username=username).exists():
            return JsonResponse({'available': False, 'message': 'Este nombre de usuario ya está en uso'})
        
        return JsonResponse({'available': True, 'message': 'Nombre de usuario disponible'})
    
    return JsonResponse({'error': 'Método no permitido'}, status=405)


def check_email_availability(request):
    """API endpoint para verificar disponibilidad de email"""
    if request.method == 'GET':
        email = request.GET.get('email', '').strip()
        
        if not email:
            return JsonResponse({'available': False, 'message': 'Email requerido'})
        
        if User.objects.filter(email=email).exists():
            return JsonResponse({'available': False, 'message': 'Este email ya está registrado'})
        
        return JsonResponse({'available': True, 'message': 'Email disponible'})
    
    return JsonResponse({'error': 'Método no permitido'}, status=405)
