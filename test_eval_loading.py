#!/usr/bin/env python3
"""
Script de prueba para verificar la carga de datos de evaluación
"""

import json
from pathlib import Path

def test_load_eval_data():
    """Prueba la carga de datos de evaluación."""
    eval_data = {
        "answerable": [],
        "unanswerable": [],
        "reports": {}
    }
    
    try:
        # Obtener la ruta del directorio del proyecto
        project_root = Path.cwd()
        print(f"Directorio del proyecto: {project_root}")
        
        # Cargar preguntas answerable
        answerable_path = project_root / "eval" / "answerable.jsonl"
        print(f"Ruta answerable: {answerable_path}")
        print(f"¿Existe? {answerable_path.exists()}")
        
        if answerable_path.exists():
            with open(answerable_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        eval_data["answerable"].append(json.loads(line))
        
        # Cargar preguntas unanswerable
        unanswerable_path = project_root / "eval" / "unanswerable.jsonl"
        print(f"Ruta unanswerable: {unanswerable_path}")
        print(f"¿Existe? {unanswerable_path.exists()}")
        
        if unanswerable_path.exists():
            with open(unanswerable_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        eval_data["unanswerable"].append(json.loads(line))
        
        # Cargar reportes
        report_answerable_path = project_root / "eval" / "report_answerable.json"
        print(f"Ruta report_answerable: {report_answerable_path}")
        print(f"¿Existe? {report_answerable_path.exists()}")
        
        if report_answerable_path.exists():
            with open(report_answerable_path, 'r', encoding='utf-8') as f:
                eval_data["reports"]["answerable"] = json.load(f)
        
        report_unanswerable_path = project_root / "eval" / "report_unanswerable.json"
        print(f"Ruta report_unanswerable: {report_unanswerable_path}")
        print(f"¿Existe? {report_unanswerable_path.exists()}")
        
        if report_unanswerable_path.exists():
            with open(report_unanswerable_path, 'r', encoding='utf-8') as f:
                eval_data["reports"]["unanswerable"] = json.load(f)
                
        print(f"\nDatos cargados:")
        print(f"- Answerable: {len(eval_data['answerable'])} preguntas")
        print(f"- Unanswerable: {len(eval_data['unanswerable'])} preguntas")
        print(f"- Reportes: {list(eval_data['reports'].keys())}")
        
        if eval_data['answerable']:
            print(f"\nPrimera pregunta answerable: {eval_data['answerable'][0]['question'][:100]}...")
        
        if eval_data['reports']:
            print(f"\nReporte answerable: {eval_data['reports'].get('answerable', {})}")
                
    except Exception as e:
        print(f"Error cargando datos de evaluación: {e}")
        import traceback
        traceback.print_exc()
    
    return eval_data

if __name__ == "__main__":
    test_load_eval_data()
