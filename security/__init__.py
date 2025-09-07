"""
Módulo de seguridad para Farmacia IA
Incluye encriptación, auditoría y utilidades de seguridad
"""

from .encryption import (
    encrypt_text,
    decrypt_text,
    encrypt_field,
    decrypt_field,
    generate_encryption_key,
    hash_user_id,
    mask_ip_address,
    EncryptionError,
)

__all__ = [
    'encrypt_text',
    'decrypt_text', 
    'encrypt_field',
    'decrypt_field',
    'generate_encryption_key',
    'hash_user_id',
    'mask_ip_address',
    'EncryptionError',
]
