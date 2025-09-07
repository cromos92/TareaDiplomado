"""
Utilidades de encriptación AES-256 para Farmacia IA
Usa cryptography para encriptación segura con nonces/salts
"""

import os
import base64
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from django.conf import settings


class EncryptionError(Exception):
    """Excepción personalizada para errores de encriptación"""
    pass


def _get_encryption_key() -> bytes:
    """
    Obtiene la clave de encriptación desde settings
    """
    key = getattr(settings, 'ENCRYPTION_KEY', None)
    if not key:
        raise EncryptionError("ENCRYPTION_KEY no está configurada en settings")
    
    try:
        # Si la clave está en base64, decodificarla
        if isinstance(key, str):
            return base64.b64decode(key)
        return key
    except Exception as e:
        raise EncryptionError(f"Error al decodificar ENCRYPTION_KEY: {e}")


def _derive_key(password: bytes, salt: bytes) -> bytes:
    """
    Deriva una clave de 32 bytes usando PBKDF2
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password)


def encrypt_text(plaintext: str) -> str:
    """
    Encripta texto usando AES-256-GCM
    
    Args:
        plaintext: Texto a encriptar
        
    Returns:
        String base64 con formato: salt:nonce:ciphertext:tag
        
    Raises:
        EncryptionError: Si hay error en la encriptación
    """
    if not plaintext:
        return ""
    
    try:
        # Generar salt y nonce aleatorios
        salt = os.urandom(16)  # 128 bits
        nonce = os.urandom(12)  # 96 bits para GCM
        
        # Derivar clave desde la master key
        master_key = _get_encryption_key()
        derived_key = _derive_key(master_key, salt)
        
        # Crear cipher AES-256-GCM
        cipher = Cipher(
            algorithms.AES(derived_key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encriptar
        ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()
        
        # Combinar salt:nonce:ciphertext:tag
        encrypted_data = salt + nonce + ciphertext + encryptor.tag
        
        # Retornar como base64
        return base64.b64encode(encrypted_data).decode('ascii')
        
    except Exception as e:
        raise EncryptionError(f"Error al encriptar: {e}")


def decrypt_text(encrypted_text: str) -> str:
    """
    Desencripta texto encriptado con encrypt_text
    
    Args:
        encrypted_text: String base64 encriptado
        
    Returns:
        Texto desencriptado
        
    Raises:
        EncryptionError: Si hay error en la desencriptación
    """
    if not encrypted_text:
        return ""
    
    try:
        # Decodificar base64
        encrypted_data = base64.b64decode(encrypted_text.encode('ascii'))
        
        # Extraer componentes
        salt = encrypted_data[:16]
        nonce = encrypted_data[16:28]
        ciphertext = encrypted_data[28:-16]
        tag = encrypted_data[-16:]
        
        # Derivar clave
        master_key = _get_encryption_key()
        derived_key = _derive_key(master_key, salt)
        
        # Crear cipher
        cipher = Cipher(
            algorithms.AES(derived_key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Desencriptar
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext.decode('utf-8')
        
    except Exception as e:
        raise EncryptionError(f"Error al desencriptar: {e}")


def generate_encryption_key() -> str:
    """
    Genera una nueva clave de encriptación de 32 bytes
    
    Returns:
        Clave en formato base64 para usar en ENCRYPTION_KEY
    """
    key = os.urandom(32)
    return base64.b64encode(key).decode('ascii')


def hash_user_id(user_id: str) -> str:
    """
    Crea un hash seguro del user_id para auditoría sin PII
    
    Args:
        user_id: ID del usuario
        
    Returns:
        Hash SHA-256 del user_id
    """
    import hashlib
    
    # Usar salt fijo para consistencia en auditoría
    salt = getattr(settings, 'USER_ID_SALT', 'farmacia_ia_audit_salt')
    
    # Crear hash
    hasher = hashlib.sha256()
    hasher.update(f"{salt}:{user_id}".encode('utf-8'))
    
    return hasher.hexdigest()


def mask_ip_address(ip_address: str) -> str:
    """
    Enmascara dirección IP para auditoría preservando utilidad
    
    Args:
        ip_address: Dirección IP original
        
    Returns:
        IP enmascarada (ej: 192.168.1.xxx)
    """
    if not ip_address:
        return ""
    
    try:
        # IPv4
        if '.' in ip_address and ip_address.count('.') == 3:
            parts = ip_address.split('.')
            return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        
        # IPv6 - mantener primeros 4 grupos
        if ':' in ip_address:
            parts = ip_address.split(':')
            if len(parts) >= 4:
                return ':'.join(parts[:4]) + '::xxxx'
        
        return "xxx.xxx.xxx.xxx"
        
    except Exception:
        return "xxx.xxx.xxx.xxx"


# Funciones de conveniencia para modelos
def encrypt_field(value: Optional[str]) -> Optional[str]:
    """Encripta un campo de modelo si no es None"""
    if value is None:
        return None
    return encrypt_text(str(value))


def decrypt_field(value: Optional[str]) -> Optional[str]:
    """Desencripta un campo de modelo si no es None"""
    if value is None:
        return None
    return decrypt_text(value)
