"""CLI: регистрирует новую версию RAG-модели в MLflow Model Registry.

Запуск:
    python -m src.stages.register_model --description "promotion to v2"

Можно также через переменные окружения:
    MLFLOW_TRACKING_URI=http://localhost:5000 \\
    MLFLOW_REGISTERED_MODEL_NAME=school-rag-assistant \\
    python -m src.stages.register_model
"""

from __future__ import annotations

import argparse
import os

from src.mlflow_registry import register_new_version


def _build_config() -> dict:
    return {
        "default_llm_model": os.getenv("DEFAULT_LLM_MODEL", "qwen2.5:7b"),
        "available_llm_models": os.getenv(
            "AVAILABLE_LLM_MODELS", "qwen2.5:7b,llama3.2"
        ),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small"
        ),
        "k_retrieval": int(os.getenv("DEFAULT_K_RETRIEVAL", "8")),
        "chroma_collection": "school_knowledge_base",
        "registered_by": os.getenv("USER", "ci"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description",
        default=None,
        help="Описание новой версии модели",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Не назначать alias после регистрации",
    )
    args = parser.parse_args()

    info = register_new_version(
        _build_config(),
        description=args.description,
        promote_alias=not args.no_promote,
    )
    print(
        f"OK: name={info.name} version={info.version} alias={info.alias} "
        f"run_id={info.run_id}"
    )


if __name__ == "__main__":
    main()
