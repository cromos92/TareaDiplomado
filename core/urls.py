"""
URL configuration for Farmacia IA project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from django.http import JsonResponse
from . import auth_views


def home_view(request):
    """Vista principal del sistema"""
    return TemplateView.as_view(template_name='base.html')(request)


def health_check(request):
    """Endpoint de health check"""
    return JsonResponse({
        'status': 'healthy',
        'service': 'farmacia-ia',
        'version': '1.0.0'
    })


urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Home
    path('', home_view, name='home'),
    
    # Health check
    path('health/', health_check, name='health'),
    
    # Autenticación
    path('auth/login/', auth_views.login_view, name='login'),
    path('auth/register/', auth_views.register_view, name='register'),
    path('auth/logout/', auth_views.logout_view, name='logout'),
    path('auth/profile/', auth_views.profile_view, name='profile'),
    path('auth/delete-account/', auth_views.delete_account_view, name='delete_account'),
    path('auth/export-data/', auth_views.export_data_view, name='export_data'),
    path('auth/check-username/', auth_views.check_username_availability, name='check_username'),
    path('auth/check-email/', auth_views.check_email_availability, name='check_email'),
    
    # Conversations app (includes API and web views)
    path('conversations/', include('conversations.urls')),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Custom error handlers
handler404 = 'core.views.handler404'
handler500 = 'core.views.handler500'