"""
Prompts y templates para los agentes de Farmacia IA
"""

from .medication_prompts import MedicationPrompts
from .pharmacy_prompts import PharmacyPrompts
from .safety_prompts import SafetyPrompts
from .emergency_prompts import EmergencyPrompts
from .clarification_prompts import ClarificationPrompts
from .response_generator_prompts import ResponseGeneratorPrompts

__all__ = [
    'MedicationPrompts',
    'PharmacyPrompts', 
    'SafetyPrompts',
    'EmergencyPrompts',
    'ClarificationPrompts',
    'ResponseGeneratorPrompts'
]
