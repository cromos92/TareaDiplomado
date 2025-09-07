"""
ASGI config for Farmacia IA project.

Configura tanto HTTP como WebSocket routing.
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator

# Configurar Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Importar routing después de configurar Django
django_asgi_app = get_asgi_application()

from conversations import routing

application = ProtocolTypeRouter({
    # HTTP requests
    "http": django_asgi_app,
    
    # WebSocket requests
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(
                routing.websocket_urlpatterns
            )
        )
    ),
})