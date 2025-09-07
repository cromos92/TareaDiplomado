"""
Cliente para la API del MINSAL (Ministerio de Salud de Chile)
Obtiene información sobre farmacias y turnos
"""

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import httpx
import redis.asyncio as redis
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)

from .normalizers import CommuneNormalizer

logger = logging.getLogger(__name__)


@dataclass
class Farmacia:
    """Estructura de datos para una farmacia"""
    id: str
    nombre: str
    direccion: str
    comuna: str
    region: str
    telefono: Optional[str] = None
    horario: Optional[str] = None
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    es_turno: bool = False
    fecha_turno: Optional[str] = None


@dataclass
class MinsalResponse:
    """Respuesta del cliente MINSAL"""
    data: List[Farmacia]
    stale: bool = False
    cached: bool = False
    timestamp: Optional[datetime] = None


class MinsalClientError(Exception):
    """Excepción base para errores del cliente MINSAL"""
    pass


class MinsalAPIError(MinsalClientError):
    """Error específico de la API MINSAL"""
    pass


class MinsalClient:
    """
    Cliente asíncrono para la API del MINSAL
    
    Proporciona acceso a información sobre farmacias y turnos
    con cache Redis y manejo de errores robusto.
    """
    
    def __init__(
        self,
        base_url: str = "https://api.minsal.cl",
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 86400,  # 24 horas
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Inicializar componentes
        self.normalizer = CommuneNormalizer()
        self._redis_client: Optional[redis.Redis] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Entrada del context manager"""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Salida del context manager"""
        await self._close_clients()
    
    async def _initialize_clients(self):
        """Inicializa los clientes HTTP y Redis"""
        try:
            # Cliente HTTP
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "User-Agent": "Farmacia-IA/1.0.0",
                    "Accept": "application/json",
                }
            )
            
            # Cliente Redis
            self._redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Verificar conexión Redis
            await self._redis_client.ping()
            
            logger.info("MINSAL client initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}. Continuing without cache.")
            self._redis_client = None
    
    async def _close_clients(self):
        """Cierra los clientes"""
        if self._http_client:
            await self._http_client.aclose()
        
        if self._redis_client:
            await self._redis_client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Realiza una request HTTP con reintentos
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de query
            
        Returns:
            Respuesta JSON de la API
            
        Raises:
            MinsalAPIError: Error en la API
        """
        if not self._http_client:
            raise MinsalClientError("HTTP client not initialized")
        
        url = urljoin(self.base_url, endpoint)
        
        try:
            logger.debug(f"Making request to {url} with params {params}")
            
            response = await self._http_client.get(url, params=params or {})
            response.raise_for_status()
            
            data = response.json()
            
            logger.debug(f"Received response with {len(data.get('data', []))} items")
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            raise MinsalAPIError(f"API error: {e.response.status_code}")
        
        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {e}")
            raise MinsalAPIError(f"Request failed: {e}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {url}")
            raise MinsalAPIError("Invalid JSON response")
    
    async def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Genera clave de cache"""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, date):
                v = v.isoformat()
            key_parts.append(f"{k}:{v}")
        
        return f"minsal:{':'.join(key_parts)}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos del cache"""
        if not self._redis_client:
            return None
        
        try:
            cached_data = await self._redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached_data)
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    async def _set_cache(self, cache_key: str, data: Dict[str, Any], ttl: int = None) -> None:
        """Guarda datos en cache"""
        if not self._redis_client:
            return
        
        try:
            cache_data = {
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": ttl or self.cache_ttl
            }
            
            await self._redis_client.setex(
                cache_key,
                ttl or self.cache_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached data for {cache_key}")
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def _get_stale_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos del cache aunque estén expirados"""
        if not self._redis_client:
            return None
        
        try:
            # Buscar claves que coincidan con el patrón
            pattern = cache_key.replace(":", "_expired_*")
            keys = await self._redis_client.keys(f"minsal:stale:{pattern}")
            
            if keys:
                # Tomar la más reciente
                latest_key = sorted(keys)[-1]
                cached_data = await self._redis_client.get(latest_key)
                
                if cached_data:
                    logger.info(f"Using stale cache for {cache_key}")
                    return json.loads(cached_data)
        
        except Exception as e:
            logger.warning(f"Stale cache read error: {e}")
        
        return None
    
    async def _save_stale_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Guarda datos como cache stale para fallback"""
        if not self._redis_client:
            return
        
        try:
            stale_key = f"minsal:stale:{cache_key}:{int(datetime.utcnow().timestamp())}"
            
            await self._redis_client.setex(
                stale_key,
                7 * 24 * 3600,  # 7 días para cache stale
                json.dumps(data, default=str)
            )
            
            # Limpiar caches stale antiguos (mantener solo los últimos 3)
            pattern = f"minsal:stale:{cache_key}:*"
            keys = await self._redis_client.keys(pattern)
            
            if len(keys) > 3:
                old_keys = sorted(keys)[:-3]
                await self._redis_client.delete(*old_keys)
        
        except Exception as e:
            logger.warning(f"Stale cache save error: {e}")
    
    def _parse_farmacia_data(self, raw_data: Dict[str, Any]) -> Farmacia:
        """Parsea datos raw de farmacia a objeto Farmacia"""
        return Farmacia(
            id=str(raw_data.get("id", "")),
            nombre=raw_data.get("nombre", "").strip(),
            direccion=raw_data.get("direccion", "").strip(),
            comuna=self.normalizer.normalize_commune(raw_data.get("comuna", "")),
            region=self.normalizer.normalize_region(raw_data.get("region", "")),
            telefono=raw_data.get("telefono"),
            horario=raw_data.get("horario"),
            latitud=self._safe_float(raw_data.get("latitud")),
            longitud=self._safe_float(raw_data.get("longitud")),
            es_turno=raw_data.get("es_turno", False),
            fecha_turno=raw_data.get("fecha_turno")
        )
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Convierte valor a float de forma segura"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    async def get_locales(self, comuna: str) -> MinsalResponse:
        """
        Obtiene farmacias de una comuna
        
        Args:
            comuna: Nombre de la comuna
            
        Returns:
            Lista de farmacias en la comuna
        """
        # Normalizar comuna
        normalized_comuna = self.normalizer.normalize_commune(comuna)
        
        # Generar clave de cache
        cache_key = await self._get_cache_key("locales", comuna=normalized_comuna)
        
        # Intentar obtener del cache
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            farmacias = [
                Farmacia(**farmacia_data) 
                for farmacia_data in cached_data["data"]
            ]
            return MinsalResponse(
                data=farmacias,
                cached=True,
                timestamp=datetime.fromisoformat(cached_data["timestamp"])
            )
        
        try:
            # Hacer request a la API
            response_data = await self._make_request(
                "/farmacias/locales",
                params={"comuna": normalized_comuna}
            )
            
            # Parsear datos
            farmacias = []
            for raw_farmacia in response_data.get("data", []):
                try:
                    farmacia = self._parse_farmacia_data(raw_farmacia)
                    farmacias.append(farmacia)
                except Exception as e:
                    logger.warning(f"Error parsing farmacia data: {e}")
                    continue
            
            # Guardar en cache
            cache_data = [asdict(farmacia) for farmacia in farmacias]
            await self._set_cache(cache_key, cache_data)
            await self._save_stale_cache(cache_key, cache_data)
            
            logger.info(f"Retrieved {len(farmacias)} farmacias for {normalized_comuna}")
            
            return MinsalResponse(
                data=farmacias,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            logger.error(f"Error getting locales for {normalized_comuna}: {e}")
            
            # Intentar obtener cache stale
            stale_data = await self._get_stale_cache(cache_key)
            if stale_data:
                farmacias = [
                    Farmacia(**farmacia_data) 
                    for farmacia_data in stale_data["data"]
                ]
                
                logger.warning(f"Using stale data for {normalized_comuna}")
                
                return MinsalResponse(
                    data=farmacias,
                    stale=True,
                    cached=True,
                    timestamp=datetime.fromisoformat(stale_data["timestamp"])
                )
            
            # Si no hay cache, devolver lista vacía
            return MinsalResponse(data=[], stale=True)
    
    async def get_locales_turno(self, comuna: str, fecha: date) -> MinsalResponse:
        """
        Obtiene farmacias de turno en una comuna para una fecha específica
        
        Args:
            comuna: Nombre de la comuna
            fecha: Fecha para consultar turnos
            
        Returns:
            Lista de farmacias de turno
        """
        # Normalizar comuna
        normalized_comuna = self.normalizer.normalize_commune(comuna)
        
        # Generar clave de cache
        cache_key = await self._get_cache_key(
            "turnos", 
            comuna=normalized_comuna, 
            fecha=fecha
        )
        
        # Intentar obtener del cache
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            farmacias = [
                Farmacia(**farmacia_data) 
                for farmacia_data in cached_data["data"]
            ]
            return MinsalResponse(
                data=farmacias,
                cached=True,
                timestamp=datetime.fromisoformat(cached_data["timestamp"])
            )
        
        try:
            # Hacer request a la API
            response_data = await self._make_request(
                "/farmacias/turnos",
                params={
                    "comuna": normalized_comuna,
                    "fecha": fecha.isoformat()
                }
            )
            
            # Parsear datos
            farmacias = []
            for raw_farmacia in response_data.get("data", []):
                try:
                    farmacia = self._parse_farmacia_data(raw_farmacia)
                    farmacia.es_turno = True
                    farmacia.fecha_turno = fecha.isoformat()
                    farmacias.append(farmacia)
                except Exception as e:
                    logger.warning(f"Error parsing turno data: {e}")
                    continue
            
            # Cache más corto para turnos (6 horas)
            cache_ttl = 6 * 3600
            
            # Guardar en cache
            cache_data = [asdict(farmacia) for farmacia in farmacias]
            await self._set_cache(cache_key, cache_data, cache_ttl)
            await self._save_stale_cache(cache_key, cache_data)
            
            logger.info(f"Retrieved {len(farmacias)} turnos for {normalized_comuna} on {fecha}")
            
            return MinsalResponse(
                data=farmacias,
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            logger.error(f"Error getting turnos for {normalized_comuna} on {fecha}: {e}")
            
            # Intentar obtener cache stale
            stale_data = await self._get_stale_cache(cache_key)
            if stale_data:
                farmacias = [
                    Farmacia(**farmacia_data) 
                    for farmacia_data in stale_data["data"]
                ]
                
                logger.warning(f"Using stale turno data for {normalized_comuna}")
                
                return MinsalResponse(
                    data=farmacias,
                    stale=True,
                    cached=True,
                    timestamp=datetime.fromisoformat(stale_data["timestamp"])
                )
            
            # Si no hay cache, devolver lista vacía
            return MinsalResponse(data=[], stale=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado del cliente MINSAL
        
        Returns:
            Estado del cliente y sus dependencias
        """
        status = {
            "minsal_client": "healthy",
            "redis_cache": "unknown",
            "api_connectivity": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Verificar Redis
        if self._redis_client:
            try:
                await self._redis_client.ping()
                status["redis_cache"] = "healthy"
            except Exception as e:
                status["redis_cache"] = f"error: {e}"
        else:
            status["redis_cache"] = "disabled"
        
        # Verificar API (con timeout corto)
        try:
            # Hacer una request simple para verificar conectividad
            await self._make_request("/health", {})
            status["api_connectivity"] = "healthy"
        except Exception as e:
            status["api_connectivity"] = f"error: {e}"
        
        return status


# Función de conveniencia para uso simple
async def get_farmacias_comuna(comuna: str, redis_url: str = None) -> List[Farmacia]:
    """
    Función de conveniencia para obtener farmacias de una comuna
    
    Args:
        comuna: Nombre de la comuna
        redis_url: URL de Redis (opcional)
        
    Returns:
        Lista de farmacias
    """
    async with MinsalClient(redis_url=redis_url or "redis://localhost:6379") as client:
        response = await client.get_locales(comuna)
        return response.data


async def get_farmacias_turno(comuna: str, fecha: date = None, redis_url: str = None) -> List[Farmacia]:
    """
    Función de conveniencia para obtener farmacias de turno
    
    Args:
        comuna: Nombre de la comuna
        fecha: Fecha (por defecto hoy)
        redis_url: URL de Redis (opcional)
        
    Returns:
        Lista de farmacias de turno
    """
    if fecha is None:
        fecha = date.today()
    
    async with MinsalClient(redis_url=redis_url or "redis://localhost:6379") as client:
        response = await client.get_locales_turno(comuna, fecha)
        return response.data
