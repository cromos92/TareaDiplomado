"""
Tests para modelos de conversations
"""

import uuid
from django.test import TestCase
from django.utils import timezone
from unittest.mock import patch
from .models import Conversation, Message, AuditLog


class ConversationModelTest(TestCase):
    """Tests para el modelo Conversation"""
    
    def setUp(self):
        self.user_id = "test_user_123"
        self.ip_address = "192.168.1.100"
    
    def test_create_conversation(self):
        """Test crear conversación básica"""
        conversation = Conversation.objects.create(
            user_id=self.user_id,
            ip_address=self.ip_address
        )
        
        self.assertIsInstance(conversation.id, uuid.UUID)
        self.assertEqual(conversation.user_id, self.user_id)
        self.assertEqual(conversation.ip_address, self.ip_address)
        self.assertIsNotNone(conversation.started_at)
        self.assertIsNone(conversation.closed_at)
        self.assertTrue(conversation.is_active)
        self.assertFalse(conversation.encrypted)
    
    def test_close_conversation(self):
        """Test cerrar conversación"""
        conversation = Conversation.objects.create(
            user_id=self.user_id,
            ip_address=self.ip_address
        )
        
        self.assertTrue(conversation.is_active)
        
        conversation.close()
        conversation.refresh_from_db()
        
        self.assertFalse(conversation.is_active)
        self.assertIsNotNone(conversation.closed_at)
    
    def test_conversation_duration(self):
        """Test cálculo de duración"""
        conversation = Conversation.objects.create(
            user_id=self.user_id,
            ip_address=self.ip_address
        )
        
        # Conversación activa debe tener duración
        duration = conversation.duration
        self.assertIsNotNone(duration)
        
        # Cerrar conversación
        conversation.close()
        duration_closed = conversation.duration
        self.assertIsNotNone(duration_closed)
    
    def test_message_count(self):
        """Test conteo de mensajes"""
        conversation = Conversation.objects.create(
            user_id=self.user_id,
            ip_address=self.ip_address
        )
        
        self.assertEqual(conversation.get_message_count(), 0)
        
        # Crear mensajes
        Message.objects.create(
            conversation=conversation,
            role='user',
            content='Test message 1'
        )
        Message.objects.create(
            conversation=conversation,
            role='assistant',
            content='Test response 1'
        )
        
        self.assertEqual(conversation.get_message_count(), 2)


class MessageModelTest(TestCase):
    """Tests para el modelo Message"""
    
    def setUp(self):
        self.conversation = Conversation.objects.create(
            user_id="test_user",
            ip_address="192.168.1.100"
        )
    
    def test_create_message(self):
        """Test crear mensaje básico"""
        message = Message.objects.create(
            conversation=self.conversation,
            role='user',
            content='Test message'
        )
        
        self.assertIsInstance(message.id, uuid.UUID)
        self.assertEqual(message.conversation, self.conversation)
        self.assertEqual(message.role, 'user')
        self.assertEqual(message.content, 'Test message')
        self.assertEqual(message.tokens, 0)
    
    def test_message_roles(self):
        """Test roles válidos de mensaje"""
        valid_roles = ['user', 'assistant', 'system']
        
        for role in valid_roles:
            message = Message.objects.create(
                conversation=self.conversation,
                role=role,
                content=f'Test {role} message'
            )
            self.assertEqual(message.role, role)


class AuditLogModelTest(TestCase):
    """Tests para el modelo AuditLog"""
    
    def setUp(self):
        self.conversation = Conversation.objects.create(
            user_id="test_user",
            ip_address="192.168.1.100"
        )
    
    def test_create_audit_log(self):
        """Test crear log de auditoría"""
        log = AuditLog.objects.create(
            event='conversation_started',
            actor='hashed_user_id',
            conversation=self.conversation,
            payload={'test': 'data'},
            ip_address='192.168.1.xxx'
        )
        
        self.assertEqual(log.event, 'conversation_started')
        self.assertEqual(log.actor, 'hashed_user_id')
        self.assertEqual(log.conversation, self.conversation)
        self.assertEqual(log.payload, {'test': 'data'})
        self.assertEqual(log.ip_address, '192.168.1.xxx')
    
    @patch('security.encryption.hash_user_id')
    @patch('security.encryption.mask_ip_address')
    def test_log_event_method(self, mock_mask_ip, mock_hash_user):
        """Test método log_event"""
        mock_hash_user.return_value = 'hashed_user'
        mock_mask_ip.return_value = '192.168.1.xxx'
        
        log = AuditLog.log_event(
            event='test_event',
            actor='user_123',
            conversation=self.conversation,
            payload={'key': 'value'},
            ip_address='192.168.1.100'
        )
        
        self.assertEqual(log.event, 'test_event')
        self.assertEqual(log.actor, 'hashed_user')
        self.assertEqual(log.conversation, self.conversation)
        self.assertEqual(log.payload, {'key': 'value'})
        self.assertEqual(log.ip_address, '192.168.1.xxx')
        
        mock_hash_user.assert_called_once_with('user_123')
        mock_mask_ip.assert_called_once_with('192.168.1.100')
