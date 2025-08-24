import argparse
import hashlib
import uuid
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
# Carga perezosa/segura de SemanticChunker para evitar fallos de import en entornos sin el módulo
try:
    from langchain_experimental.text_splitters import SemanticChunker as _SemanticChunker
except Exception:  # pragma: no cover
    _SemanticChunker = None  # type: ignore
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient


def _stable_id(source: str, page: int, chunk_index: int, content: str) -> str:
    # Qdrant solo acepta IDs unsigned integer o UUID.
    # Generamos un UUID v5 determinístico a partir de los atributos del chunk.
    name = f"{source}|{page}|{chunk_index}|{len(content)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _load_documents(data_dir: Path, include_patterns: List[str]) -> List:
    docs = []
    for pattern in include_patterns:
        for file_path in sorted(data_dir.rglob(pattern)):
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs.extend(loader.load())
            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs.extend(loader.load())
            elif file_path.suffix.lower() == ".doc":
                print(
                    f"[Aviso] Se encontró archivo .doc legacy: {file_path}. "
                    "Convierte a .docx para ingestar (o habilita un loader alternativo)."
                )
            else:
                # Fallback plain text loader (handles .txt/.md/.rst)
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs.extend(loader.load())
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta de documentos a Qdrant usando OpenAI Embeddings y LangChain"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directorio con documentos a indexar (pdf/txt)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default="*.pdf,*.txt,*.docx",
        help="Patrones separados por coma a incluir (e.g. '*.pdf,*.txt,*.docx,*.md')",
    )
    parser.add_argument("--chunker", type=str, choices=["recursive", "semantic"], default="recursive")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--semantic-threshold-type",
        type=str,
        choices=["percentile", "standard_deviation"],
        default="percentile",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=95,
        help="Umbral para SemanticChunker (percentil o desviación estándar)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Modelo de embeddings de OpenAI",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Sobrescribe QDRANT_COLLECTION del entorno si se provee",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra información de la colección y conteo tras la ingesta",
    )
    args = parser.parse_args()

    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = args.collection or os.getenv("QDRANT_COLLECTION")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not qdrant_url or not qdrant_api_key or not qdrant_collection:
        raise SystemExit(
            "Faltan variables de entorno: QDRANT_URL, QDRANT_API_KEY y/o QDRANT_COLLECTION"
        )
    if not openai_api_key:
        raise SystemExit("Falta OPENAI_API_KEY para generar embeddings")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"El directorio de datos no existe: {data_dir}")

    include_patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    raw_docs = _load_documents(data_dir, include_patterns)
    if not raw_docs:
        raise SystemExit(
            f"No se encontraron documentos en {data_dir} con patrones {include_patterns}"
        )

    # Mostrar info de embeddings
    embed_dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(
        args.embedding_model, "desconocido"
    )
    print(
        f"Usando embeddings '{args.embedding_model}' (dimensiones: {embed_dims}). "
        f"Chunker: {args.chunker}"
    )

    embeddings = OpenAIEmbeddings(model=args.embedding_model)

    if args.chunker == "semantic":
        if _SemanticChunker is None:
            print(
                "[Aviso] SemanticChunker no disponible en este entorno. "
                "Se usará chunking 'recursive'. Puedes instalar/actualizar langchain-experimental."
            )
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
            )
        else:
            splitter = _SemanticChunker(
                embeddings,
                breakpoint_threshold_type=args.semantic_threshold_type,
                breakpoint_threshold_amount=args.semantic_threshold,
            )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

    splits = splitter.split_documents(raw_docs)

    # IDs estables por chunk
    ids = []
    for idx, d in enumerate(splits):
        # Enriquecer metadatos
        source = d.metadata.get("source", "unknown")
        file_name = os.path.basename(source)
        file_ext = os.path.splitext(file_name)[1].lower()
        doc_type = (
            "pdf"
            if file_ext == ".pdf"
            else "word" if file_ext in {".doc", ".docx"} else "text"
        )
        page = int(d.metadata.get("page", 0))
        d.metadata.update(
            {
                "source": source,
                "file_name": file_name,
                "file_ext": file_ext,
                "doc_type": doc_type,
                "page": page,
                "chunk_index": idx,
            }
        )
        ids.append(_stable_id(source, page, idx, d.page_content))

    print(
        f"Subiendo {len(splits)} chunks a Qdrant colección '{qdrant_collection}' en {qdrant_url}..."
    )

    # Crea la colección si no existe y hace upsert
    _ = Qdrant.from_documents(
        splits,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=qdrant_collection,
        ids=ids,
    )

    print("Ingesta completada.")

    if args.show:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            info = client.get_collection(qdrant_collection)
            count = client.count(qdrant_collection, exact=True).count
            print("Colección:", qdrant_collection)
            print("Vector size:", info.config.params.vectors.size)
            print("Distance:", getattr(info.config.params.vectors, "distance", "cosine"))
            print("Total points:", count)
        except Exception as e:
            print("No se pudo obtener info de la colección:", e)


if __name__ == "__main__":
    main()


