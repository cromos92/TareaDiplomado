"""
Vistas principales para Farmacia IA
"""

from django.shortcuts import render
from django.http import JsonResponse


def handler404(request, exception):
    """Maneja errores 404"""
    if request.path.startswith('/api/'):
        return JsonResponse({
            'error': 'Not Found',
            'detail': 'The requested resource was not found.'
        }, status=404)
    
    return render(request, 'errors/404.html', status=404)


def handler500(request):
    """Maneja errores 500"""
    if request.path.startswith('/api/'):
        return JsonResponse({
            'error': 'Internal Server Error',
            'detail': 'An internal server error occurred.'
        }, status=500)
    
    return render(request, 'errors/500.html', status=500)
