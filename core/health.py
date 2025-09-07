"""
Health check views for Railway deployment
"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint for Railway
    """
    return JsonResponse({
        "status": "healthy",
        "service": "farmacia-ia-django",
        "version": "1.0.0"
    })


@csrf_exempt
@require_http_methods(["GET"])
def readiness_check(request):
    """
    Readiness check endpoint
    """
    # Add checks for database, redis, etc.
    try:
        from django.db import connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return JsonResponse({
        "status": "ready" if db_status == "healthy" else "not_ready",
        "checks": {
            "database": db_status
        }
    })
