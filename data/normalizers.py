"""
Normalizadores para datos geográficos de Chile
"""

import re
from typing import Dict, Optional
from unicodedata import normalize


class CommuneNormalizer:
    """
    Normalizador de nombres de comunas y regiones de Chile
    """
    
    def __init__(self):
        # Mapeo de comunas con variaciones comunes
        self.commune_mapping = {
            # Región de Antofagasta
            "antofagasta": "Antofagasta",
            "calama": "Calama",
            "tocopilla": "Tocopilla",
            "mejillones": "Mejillones",
            "taltal": "Taltal",
            "san pedro de atacama": "San Pedro de Atacama",
            "san pedro atacama": "San Pedro de Atacama",
            "ollague": "Ollagüe",
            "maria elena": "María Elena",
            "maría elena": "María Elena",
            
            # Santiago
            "santiago": "Santiago",
            "las condes": "Las Condes",
            "providencia": "Providencia",
            "ñuñoa": "Ñuñoa",
            "nunoa": "Ñuñoa",
            "la reina": "La Reina",
            "peñalolen": "Peñalolén",
            "penalolen": "Peñalolén",
            "vitacura": "Vitacura",
            "lo barnechea": "Lo Barnechea",
            "huechuraba": "Huechuraba",
            "quilicura": "Quilicura",
            "renca": "Renca",
            "pudahuel": "Pudahuel",
            "maipu": "Maipú",
            "maipú": "Maipú",
            "cerrillos": "Cerrillos",
            "estacion central": "Estación Central",
            "estación central": "Estación Central",
            "pedro aguirre cerda": "Pedro Aguirre Cerda",
            "san miguel": "San Miguel",
            "san joaquin": "San Joaquín",
            "san joaquín": "San Joaquín",
            "macul": "Macul",
            "la cisterna": "La Cisterna",
            "san ramon": "San Ramón",
            "san ramón": "San Ramón",
            "la granja": "La Granja",
            "la pintana": "La Pintana",
            "puente alto": "Puente Alto",
            "san bernardo": "San Bernardo",
            "el bosque": "El Bosque",
            "la florida": "La Florida",
            "independencia": "Independencia",
            "recoleta": "Recoleta",
            "conchali": "Conchalí",
            "conchalí": "Conchalí",
            
            # Valparaíso
            "valparaiso": "Valparaíso",
            "valparaíso": "Valparaíso",
            "viña del mar": "Viña del Mar",
            "vina del mar": "Viña del Mar",
            "con con": "Con Con",
            "concón": "Concón",
            "concon": "Concón",
            "quilpue": "Quilpué",
            "villa alemana": "Villa Alemana",
            "limache": "Limache",
            "olmue": "Olmué",
            "olmué": "Olmué",
            "quillota": "Quillota",
            "la calera": "La Calera",
            "hijuelas": "Hijuelas",
            "nogales": "Nogales",
            "san antonio": "San Antonio",
            "cartagena": "Cartagena",
            "el tabo": "El Tabo",
            "el quisco": "El Quisco",
            "algarrobo": "Algarrobo",
            "casablanca": "Casablanca",
            
            # Otras comunas importantes
            "temuco": "Temuco",
            "concepcion": "Concepción",
            "concepción": "Concepción",
            "talcahuano": "Talcahuano",
            "chillan": "Chillán",
            "chillán": "Chillán",
            "los angeles": "Los Ángeles",
            "los ángeles": "Los Ángeles",
            "rancagua": "Rancagua",
            "talca": "Talca",
            "curico": "Curicó",
            "curicó": "Curicó",
            "linares": "Linares",
            "valdivia": "Valdivia",
            "osorno": "Osorno",
            "puerto montt": "Puerto Montt",
            "castro": "Castro",
            "ancud": "Ancud",
            "coyhaique": "Coyhaique",
            "puerto aysén": "Puerto Aysén",
            "puerto aysen": "Puerto Aysén",
            "punta arenas": "Punta Arenas",
            "puerto natales": "Puerto Natales",
            "porvenir": "Porvenir",
            "arica": "Arica",
            "iquique": "Iquique",
            "alto hospicio": "Alto Hospicio",
            "copiapo": "Copiapó",
            "copiapó": "Copiapó",
            "la serena": "La Serena",
            "coquimbo": "Coquimbo",
            "ovalle": "Ovalle",
            "illapel": "Illapel",
        }
        
        # Mapeo de regiones
        self.region_mapping = {
            "arica y parinacota": "Arica y Parinacota",
            "tarapaca": "Tarapacá",
            "tarapacá": "Tarapacá",
            "antofagasta": "Antofagasta",
            "atacama": "Atacama",
            "coquimbo": "Coquimbo",
            "valparaiso": "Valparaíso",
            "valparaíso": "Valparaíso",
            "metropolitana": "Metropolitana de Santiago",
            "metropolitana de santiago": "Metropolitana de Santiago",
            "rm": "Metropolitana de Santiago",
            "ohiggins": "O'Higgins",
            "o'higgins": "O'Higgins",
            "maule": "Maule",
            "ñuble": "Ñuble",
            "nuble": "Ñuble",
            "biobio": "Biobío",
            "biobío": "Biobío",
            "araucania": "Araucanía",
            "araucanía": "Araucanía",
            "los rios": "Los Ríos",
            "los ríos": "Los Ríos",
            "los lagos": "Los Lagos",
            "aysen": "Aysén del General Carlos Ibáñez del Campo",
            "aysén": "Aysén del General Carlos Ibáñez del Campo",
            "magallanes": "Magallanes y de la Antártica Chilena",
            "magallanes y antartica": "Magallanes y de la Antártica Chilena",
            "magallanes y antártica": "Magallanes y de la Antártica Chilena",
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto removiendo acentos y convirtiendo a minúsculas
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        # Remover acentos y normalizar
        normalized = normalize('NFD', text.lower().strip())
        normalized = ''.join(c for c in normalized if not c.isdecimal() and c.isprintable())
        
        # Limpiar espacios múltiples
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def normalize_commune(self, commune: str) -> str:
        """
        Normaliza nombre de comuna
        
        Args:
            commune: Nombre de la comuna
            
        Returns:
            Nombre normalizado de la comuna
        """
        if not commune:
            return ""
        
        # Normalizar texto
        normalized = self.normalize_text(commune)
        
        # Buscar en mapeo
        if normalized in self.commune_mapping:
            return self.commune_mapping[normalized]
        
        # Si no está en el mapeo, capitalizar palabras
        return self._capitalize_words(commune.strip())
    
    def normalize_region(self, region: str) -> str:
        """
        Normaliza nombre de región
        
        Args:
            region: Nombre de la región
            
        Returns:
            Nombre normalizado de la región
        """
        if not region:
            return ""
        
        # Normalizar texto
        normalized = self.normalize_text(region)
        
        # Buscar en mapeo
        if normalized in self.region_mapping:
            return self.region_mapping[normalized]
        
        # Si no está en el mapeo, capitalizar palabras
        return self._capitalize_words(region.strip())
    
    def _capitalize_words(self, text: str) -> str:
        """
        Capitaliza palabras respetando artículos y preposiciones
        
        Args:
            text: Texto a capitalizar
            
        Returns:
            Texto con capitalización correcta
        """
        if not text:
            return ""
        
        # Palabras que no se capitalizan (excepto al inicio)
        lowercase_words = {'de', 'del', 'la', 'las', 'el', 'los', 'y', 'e'}
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in lowercase_words:
                # Capitalizar primera letra
                result.append(word.capitalize())
            else:
                # Mantener en minúscula
                result.append(word.lower())
        
        return ' '.join(result)
    
    def get_commune_variations(self, commune: str) -> list:
        """
        Obtiene variaciones posibles de un nombre de comuna
        
        Args:
            commune: Nombre de la comuna
            
        Returns:
            Lista de variaciones posibles
        """
        normalized = self.normalize_commune(commune)
        variations = [normalized]
        
        # Agregar variación sin acentos
        no_accents = self.normalize_text(normalized)
        if no_accents != normalized.lower():
            variations.append(no_accents)
        
        # Agregar variaciones comunes
        for key, value in self.commune_mapping.items():
            if value == normalized and key not in variations:
                variations.append(key)
        
        return list(set(variations))
    
    def is_valid_commune(self, commune: str) -> bool:
        """
        Verifica si una comuna es válida
        
        Args:
            commune: Nombre de la comuna
            
        Returns:
            True si la comuna es válida
        """
        if not commune:
            return False
        
        normalized = self.normalize_text(commune)
        return normalized in self.commune_mapping
    
    def search_communes(self, query: str, limit: int = 10) -> list:
        """
        Busca comunas que coincidan con una consulta
        
        Args:
            query: Consulta de búsqueda
            limit: Límite de resultados
            
        Returns:
            Lista de comunas que coinciden
        """
        if not query:
            return []
        
        query_normalized = self.normalize_text(query)
        matches = []
        
        for key, value in self.commune_mapping.items():
            if query_normalized in key or query_normalized in self.normalize_text(value):
                matches.append(value)
        
        # Remover duplicados y limitar
        unique_matches = list(set(matches))
        return unique_matches[:limit]


# Instancia global para uso conveniente
commune_normalizer = CommuneNormalizer()