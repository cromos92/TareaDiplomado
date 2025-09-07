"""
Comando de Django para verificar la configuración de LangSmith
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from core.langsmith_config import langsmith_config, log_agent_metrics
import time


class Command(BaseCommand):
    help = 'Verifica la configuración de LangSmith y ejecuta un test básico'

    def add_arguments(self, parser):
        parser.add_argument(
            '--run-test',
            action='store_true',
            help='Ejecuta un test de tracing básico',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🔍 Verificando configuración de LangSmith...')
        )
        
        # Verificar variables de entorno
        self.stdout.write('\n📋 Variables de entorno:')
        self.stdout.write(f'  LANGCHAIN_API_KEY: {"✅ Configurada" if settings.LANGCHAIN_API_KEY else "❌ No configurada"}')
        self.stdout.write(f'  LANGCHAIN_ENDPOINT: {settings.LANGCHAIN_ENDPOINT}')
        self.stdout.write(f'  LANGCHAIN_PROJECT: {settings.LANGCHAIN_PROJECT}')
        self.stdout.write(f'  LANGCHAIN_TRACING_V2: {"✅ Habilitado" if settings.LANGCHAIN_TRACING_V2 else "❌ Deshabilitado"}')
        self.stdout.write(f'  LANGCHAIN_TIMEOUT: {settings.LANGCHAIN_TIMEOUT}s')
        
        # Verificar configuración de LangSmith
        self.stdout.write('\n🔧 Estado de LangSmith:')
        if langsmith_config.is_enabled:
            self.stdout.write(self.style.SUCCESS('  ✅ LangSmith habilitado y configurado'))
            self.stdout.write(f'  📊 Proyecto: {langsmith_config.project}')
            self.stdout.write(f'  🌐 Endpoint: {langsmith_config.endpoint}')
        else:
            self.stdout.write(self.style.WARNING('  ⚠️  LangSmith no habilitado'))
            if not settings.LANGCHAIN_API_KEY:
                self.stdout.write('    - Falta LANGCHAIN_API_KEY')
            if not settings.LANGCHAIN_TRACING_V2:
                self.stdout.write('    - LANGCHAIN_TRACING_V2 no está habilitado')
        
        # Ejecutar test si se solicita
        if options['run_test']:
            self.run_langsmith_test()
    
    def run_langsmith_test(self):
        """Ejecuta un test básico de LangSmith"""
        self.stdout.write('\n🧪 Ejecutando test de LangSmith...')
        
        if not langsmith_config.is_enabled:
            self.stdout.write(
                self.style.WARNING('⚠️  No se puede ejecutar el test - LangSmith no habilitado')
            )
            return
        
        try:
            # Test básico de logging de métricas
            start_time = time.time()
            
            # Simular procesamiento de agente
            time.sleep(0.1)  # Simular trabajo
            
            execution_time = time.time() - start_time
            
            # Log métricas de test
            log_agent_metrics(
                agent_name="TestAgent",
                input_data={
                    "test_message": "Hola, esto es un test de LangSmith",
                    "user_id": "test_user",
                    "trace_id": "test_trace_123"
                },
                output_data={
                    "response": "Test completado exitosamente",
                    "status": "success",
                    "execution_time": execution_time
                },
                execution_time=execution_time,
                success=True
            )
            
            self.stdout.write(
                self.style.SUCCESS('✅ Test de LangSmith completado exitosamente')
            )
            self.stdout.write(f'  ⏱️  Tiempo de ejecución: {execution_time:.3f}s')
            self.stdout.write(f'  🔗 Revisa el proyecto en: {langsmith_config.endpoint}')
            
        except ImportError:
            self.stdout.write(
                self.style.ERROR('❌ Error: langsmith no está instalado')
            )
            self.stdout.write('   Instala con: pip install langsmith')
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error en test de LangSmith: {e}')
            )
            self.stdout.write('   Verifica tu API key y configuración')
