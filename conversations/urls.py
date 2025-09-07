"""
URLs para la app conversations
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .api import ChatAPIView, ConversationViewSet, MessageViewSet

# Router para DRF
router = DefaultRouter()
router.register(r'conversations', ConversationViewSet, basename='conversation')
router.register(r'messages', MessageViewSet, basename='message')

app_name = 'conversations'

urlpatterns = [
    # API endpoints
    path('api/chat/', ChatAPIView.as_view(), name='chat-api'),
    path('api/', include(router.urls)),
    
    # Web views
    path('', views.conversation_list, name='conversation-list'),
    path('chat/', views.chat_view, name='chat'),
    path('chat/<uuid:conversation_id>/', views.chat_view, name='chat-with-id'),
    path('<uuid:conversation_id>/', views.conversation_detail, name='conversation-detail'),
    path('<uuid:conversation_id>/messages/', views.conversation_messages, name='conversation-messages'),
    path('new/', views.new_conversation, name='new-conversation'),
]
