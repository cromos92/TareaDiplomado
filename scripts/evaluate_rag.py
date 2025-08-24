import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_local(answerable_path: Path, unanswerable_path: Path) -> Dict[str, Any]:
    # Evaluación local directamente contra la cadena remota de langserve vía HTTP s
    # (más simple que importar server internamente; evita conflictos de entornos) OK
    import httpx

    base = os.getenv("RAG_BASE_URL", "http://localhost:8000")
    invoke_url = base.rstrip("/") + "/rag/invoke"

    ans = load_jsonl(answerable_path)
    uans = load_jsonl(unanswerable_path)

    def ask(q: str) -> str:
        t0 = time.time()
        with httpx.Client(timeout=60) as client:
            r = client.post(invoke_url, json={"input": {"question": q}})
            r.raise_for_status()
            data = r.json()
        latency = time.time() - t0
        return data.get("output", ""), latency

    total_lat = 0.0
    # Métricas answerable: conteo de respuestas no vacías
    ans_ok = 0
    for row in ans:
        out, lat = ask(row["question"])
        total_lat += lat
        if out and out.strip():
            ans_ok += 1

    # Métricas unanswerable: abstención correcta
    abst_phrase = os.getenv("RAG_ABSTENTION_TEXT", "No tengo información suficiente").lower()
    uans_ok = 0
    for row in uans:
        out, lat = ask(row["question"])
        total_lat += lat
        if abst_phrase in (out or "").lower():
            uans_ok += 1

    n = len(ans) + len(uans)
    avg_lat = total_lat / max(1, n)
    return {
        "answerable_count": len(ans),
        "answerable_nonempty": ans_ok,
        "answerable_rate": round(ans_ok / max(1, len(ans)), 3),
        "unanswerable_count": len(uans),
        "unanswerable_abstentions": uans_ok,
        "unanswerable_abstention_rate": round(uans_ok / max(1, len(uans)), 3),
        "avg_latency_s": round(avg_lat, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación simple de RAG vía /rag/invoke")
    parser.add_argument("--answerable", type=str, default="eval/answerable.jsonl")
    parser.add_argument("--unanswerable", type=str, default="eval/unanswerable.jsonl")
    parser.add_argument("--report", type=str, default="eval/report.json")
    args = parser.parse_args()

    load_dotenv()

    report = eval_local(Path(args.answerable), Path(args.unanswerable))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


