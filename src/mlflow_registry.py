"""Работа с MLflow Model Registry для RAG-ассистента.

Модель здесь - это конфигурация RAG-пайплайна (LLM + embeddings + retriever +
prompt). Мы заворачиваем её в `mlflow.pyfunc.PythonModel`, регистрируем в
Model Registry под именем `MLFLOW_REGISTERED_MODEL_NAME` и продвигаем
последнюю версию по alias (например `production`).

API:
    register_new_version(...) -> ModelVersionInfo
    get_active_version(...) -> ModelVersionInfo | None
    list_versions(...) > list[ModelVersionInfo]
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "school-rag-assistant"
DEFAULT_ALIAS = "production"


@dataclass
class ModelVersionInfo:
    name: str
    version: str
    alias: Optional[str]
    run_id: Optional[str]
    source: Optional[str]
    creation_time: Optional[int]
    tags: dict


class _RagPyFuncModel(mlflow.pyfunc.PythonModel):
    """Обёртка над RAG-конфигом
    Реальный inference выполняется в FastAPI
    """

    def load_context(self, context):
        import json
        from pathlib import Path

        cfg_path = Path(context.artifacts["config"])
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))

    def predict(self, context, model_input, params=None):
        return [self.config for _ in range(len(model_input))]


def _model_name() -> str:
    return os.getenv("MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_MODEL_NAME)


def _alias() -> str:
    return os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_ALIAS)


def _client() -> MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def register_new_version(
    config: dict[str, Any],
    *,
    description: Optional[str] = None,
    promote_alias: bool = True,
) -> ModelVersionInfo:
    """Сохраняет конфигурацию RAG-пайплайна как новую версию модели в Registry"""
    import json
    import tempfile
    from pathlib import Path

    name = _model_name()
    alias = _alias()
    client = _client()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "School_RAG_System")
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "rag_config.json"
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))

        with mlflow.start_run(run_name=f"register_{name}") as run:
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

            model_info = mlflow.pyfunc.log_model(
                artifact_path="rag_model",
                python_model=_RagPyFuncModel(),
                artifacts={"config": str(cfg_path)},
                registered_model_name=name,
            )
            run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"Модель {name} не появилась в Registry после регистрации")
    latest = max(versions, key=lambda v: int(v.version))

    if description:
        client.update_model_version(
            name=name, version=latest.version, description=description
        )

    if promote_alias:
        client.set_registered_model_alias(
            name=name, alias=alias, version=latest.version
        )

    return ModelVersionInfo(
        name=name,
        version=latest.version,
        alias=alias if promote_alias else None,
        run_id=run_id,
        source=getattr(model_info, "model_uri", latest.source),
        creation_time=latest.creation_timestamp,
        tags=dict(latest.tags or {}),
    )


def get_active_version() -> Optional[ModelVersionInfo]:
    """Возвращает версию, на которую указывает alias (по умолчанию production)"""
    name = _model_name()
    alias = _alias()
    client = _client()

    try:
        mv = client.get_model_version_by_alias(name=name, alias=alias)
    except MlflowException:
        return None

    return ModelVersionInfo(
        name=mv.name,
        version=mv.version,
        alias=alias,
        run_id=mv.run_id,
        source=mv.source,
        creation_time=mv.creation_timestamp,
        tags=dict(mv.tags or {}),
    )


def list_versions(limit: int = 20) -> list[ModelVersionInfo]:
    name = _model_name()
    alias = _alias()
    client = _client()

    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        return []

    active_version: Optional[str] = None
    try:
        active_version = client.get_model_version_by_alias(
            name=name, alias=alias
        ).version
    except MlflowException:
        active_version = None

    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)[
        :limit
    ]
    result: list[ModelVersionInfo] = []
    for v in versions_sorted:
        result.append(
            ModelVersionInfo(
                name=v.name,
                version=v.version,
                alias=alias if v.version == active_version else None,
                run_id=v.run_id,
                source=v.source,
                creation_time=v.creation_timestamp,
                tags=dict(v.tags or {}),
            )
        )
    return result


def info_to_dict(info: Optional[ModelVersionInfo]) -> Optional[dict]:
    return asdict(info) if info else None
