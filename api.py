import csv
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import mlflow
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from src.mlflow_registry import (
    get_active_version,
    info_to_dict,
    list_versions,
    register_new_version,
)
from src.monitoring.drift_detector import ConceptDriftDetector, MinimalDriftDetector

CHROMA_DIR = "chroma_langchain_db"
FEEDBACK_CSV = "data/models/feedback.csv"
AVAILABLE_MODELS = ["qwen2.5:7b", "llama3.2"]
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5:7b")
DEFAULT_K = int(os.getenv("DEFAULT_K_RETRIEVAL", "8"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

TEMPLATE = """Вы — экспертный аналитик базы знаний школы.
Ваша цель: найти ответ на вопрос в предоставленных фрагментах документов.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ИНСТРУКЦИЯ:
1. Проанализируй контекст. Если информация представлена в виде списка,
таблицы или расписания — изучи каждую строку.
2. Если в тексте упоминаются похожие термины (например, "питание" вместо "завтрак"),
используй их для ответа.
3. Если ответ найден частично, напиши то, что удалось найти.
4. Сначала кратко опиши, что ты нашел в документах, а затем дай итоговый ответ.

ОТВЕТ:"""

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
_mlflow_exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "School_RAG_System")
try:
    mlflow.set_experiment(_mlflow_exp)
except Exception:
    pass


@contextmanager
def _optional_mlflow_run(run_name: str):
    """Контекст MLflow run. yield True — активен run, можно вызывать mlflow.log_*."""
    if os.getenv("MLFLOW_DISABLED", "").lower() in ("1", "true", "yes"):
        yield False
        return
    try:
        with mlflow.start_run(run_name=run_name):
            yield True
    except Exception:
        yield False


RAG_FEEDBACK_RATING_TOTAL = Counter(
    "rag_feedback_rating_total",
    "Отзывы пользователей по оценке rating (0 — дизлайк, 1 — лайк).",
    ("rating",),
)

app = FastAPI(
    title="School RAG API",
    description="API для RAG-системы школьного ИИ-ассистента",
    version="1.1.0",
)
drift_detector = MinimalDriftDetector()
concept_detector = ConceptDriftDetector()
_vector_store = None
_active_model_version: Optional[dict] = None

CONCEPT_ALERTS_FILE = Path("data/monitoring/concept_alerts.json")
ALERTS_FILE = Path("data/monitoring/alerts.json")
ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _save_alert_to_file(result: dict) -> None:
    """Сохраняет алерт в JSON для отображения в Streamlit"""
    alerts = []
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []

    alerts.insert(0, result)
    alerts = alerts[:50]

    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False, default=str)


def _save_concept_alert_to_file(result: dict) -> None:
    """Сохраняет concept-алерт в JSON для отображения в Streamlit"""
    alerts = []
    if CONCEPT_ALERTS_FILE.exists():
        try:
            with open(CONCEPT_ALERTS_FILE, "r", encoding="utf-8") as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []
    alerts.insert(0, result)
    alerts = alerts[:50]
    with open(CONCEPT_ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False, default=str)


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )
        _vector_store = Chroma(
            collection_name="school_knowledge_base",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vector_store


def _current_rag_config() -> dict:
    """Снимок конфигурации RAG-пайплайна — то, что мы регистрируем как версию модели."""
    return {
        "default_llm_model": DEFAULT_MODEL,
        "available_llm_models": ",".join(AVAILABLE_MODELS),
        "embedding_model": EMBEDDING_MODEL,
        "k_retrieval": DEFAULT_K,
        "chroma_collection": "school_knowledge_base",
        "prompt_template_hash": str(hash(TEMPLATE)),
        "api_version": app.version,
    }


def _refresh_active_model_version() -> Optional[dict]:
    """Подгружает активную версию модели из MLflow Registry в кеш."""
    global _active_model_version
    try:
        info = get_active_version()
    except Exception as e:
        print(f"[mlflow] не удалось получить активную версию модели: {e}")
        return _active_model_version
    _active_model_version = info_to_dict(info)
    return _active_model_version


def _compute_faithfulness(answer: str, context: str) -> float:
    """Доля слов ответа найденных в контексте"""
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def _compute_answer_relevancy(question: str, answer: str) -> float:
    """Косинусное сходство между вопросом и ответом"""
    embeddings = get_vector_store()._embedding_function
    emb_q = embeddings.embed_query(question)
    emb_a = embeddings.embed_query(answer)
    return float(cosine_similarity([emb_q], [emb_a])[0][0])


# СХЕМЫ


class AskRequest(BaseModel):
    question: str
    model: Optional[str] = DEFAULT_MODEL
    k_retrieval: Optional[int] = DEFAULT_K


class SourceItem(BaseModel):
    source: str
    content: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[SourceItem]
    latency: float
    model: str
    model_version: Optional[str] = None


class FeedbackRequest(BaseModel):
    request_id: str
    question: str
    answer: str
    rating: int  # 1 - лайк, 0 - дизлайк
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


class RegisterModelRequest(BaseModel):
    description: Optional[str] = None
    promote_alias: bool = True


# ЭНДПОИНТЫ
@app.on_event("startup")
def _on_startup() -> None:
    if os.getenv("MLFLOW_AUTO_REGISTER_ON_STARTUP", "0").lower() in (
        "1",
        "true",
        "yes",
    ):
        try:
            info = register_new_version(
                _current_rag_config(),
                description="Авторегистрация версии при старте сервиса",
            )
            globals()["_active_model_version"] = info_to_dict(info)
            print(f"[mlflow] зарегистрирована версия {info.name} v{info.version}")
        except Exception as e:
            print(f"[mlflow] не удалось зарегистрировать модель при старте: {e}")
            _refresh_active_model_version()
    else:
        _refresh_active_model_version()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "School RAG API",
        "model_version": _active_model_version,
    }


@app.get("/models/active")
def models_active():
    """Текущая версия модели, на которую указывает alias (production)."""
    info = _refresh_active_model_version()
    if not info:
        raise HTTPException(
            status_code=404,
            detail="В MLflow Registry нет активной версии. "
            "Вызовите POST /models/register.",
        )
    return info


@app.get("/models/versions")
def models_versions(limit: int = 20):
    """Список последних версий зарегистрированной модели."""
    versions = [info_to_dict(v) for v in list_versions(limit=limit)]
    return {"count": len(versions), "versions": versions}


@app.post("/models/register")
def models_register(req: RegisterModelRequest):
    """Регистрирует новую версию модели в MLflow Model Registry."""
    try:
        info = register_new_version(
            _current_rag_config(),
            description=req.description,
            promote_alias=req.promote_alias,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow registry error: {e}")
    globals()["_active_model_version"] = info_to_dict(info)
    return _active_model_version


@app.get("/monitoring/drift")
def check_drift(hours: int = 24, test_mode: bool = False):
    result = drift_detector.detect_drift(hours=hours, test_mode=test_mode)

    if result.get("drift_detected"):
        print(f"DRIFT DETECTED: {result['drift_score']} > {result['threshold']}")
        _save_alert_to_file(result)

    return result


@app.get("/monitoring/alerts")
def get_alerts():
    """Для Streamlit: возвращает последние алерты"""
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


@app.get("/monitoring/concept-drift")
def check_concept_drift(hours: int = 24):
    result = concept_detector.detect_concept_drift(hours=hours)
    if result.get("concept_drift_detected"):
        print(f"CONCEPT DRIFT DETECTED: {result['issues']}")
        _save_concept_alert_to_file(result)
    return result


@app.get("/monitoring/concept-alerts")
def get_concept_alerts():
    if CONCEPT_ALERTS_FILE.exists():
        try:
            with open(CONCEPT_ALERTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


@app.post("/monitoring/seed-test-data")
def seed_test_data(n_reference: int = 30, n_current: int = 15):
    """Только для тестирования. Генерирует данные с дрейфом."""
    result = drift_detector.seed_test_data(n_reference, n_current)
    return {"status": "ok", **result}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{request.model}' недоступна. "
            f"Доступные: {AVAILABLE_MODELS}",
        )

    request_id = str(uuid.uuid4())
    active_version = _active_model_version or {}
    model_version_str = active_version.get("version") if active_version else None

    with _optional_mlflow_run(run_name=f"API_Query_{time.strftime('%H%M%S')}") as _mf:
        try:
            start_time = time.time()

            if _mf:
                mlflow.log_param("request_id", request_id)
                mlflow.log_param("model", request.model)
                mlflow.log_param("k_retrieval", request.k_retrieval)
                mlflow.log_param("question", request.question)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL)
                if model_version_str:
                    mlflow.log_param("registered_model_version", model_version_str)
                if active_version.get("name"):
                    mlflow.log_param("registered_model_name", active_version["name"])
                    mlflow.set_tag(
                        "mlflow.modelVersion",
                        f"{active_version['name']}:{model_version_str}",
                    )

            vector_store = get_vector_store()
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            model = OllamaLLM(
                model=request.model,
                temperature=0.1,
                base_url=ollama_url,
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": request.k_retrieval}
            )

            prompt = ChatPromptTemplate.from_template(TEMPLATE)
            chain = prompt | model

            docs = retriever.invoke(request.question)
            context_text = "\n\n".join(
                [
                    f"[Источник: {d.metadata.get('source', 'Неизвестно')}]"
                    f"\n{d.page_content}"
                    for d in docs
                ]
            )

            answer = chain.invoke(
                {"context": context_text, "question": request.question}
            )
            latency = time.time() - start_time

            if _mf:
                mlflow.log_metric("latency", latency)
                mlflow.log_metric("context_length", len(context_text))
                mlflow.log_text(answer, "assistant_response.txt")

            try:
                faith = _compute_faithfulness(answer, context_text)
                relevancy = _compute_answer_relevancy(request.question, answer)

                concept_detector.log_quality_score(
                    request_id=request_id,
                    question=request.question,
                    faithfulness=faith,
                    answer_relevancy=relevancy,
                )
                if _mf:
                    mlflow.log_metric("faithfulness", faith)
                    mlflow.log_metric("answer_relevancy", relevancy)
            except Exception as quality_error:
                print(f"Quality scoring failed: {quality_error}")

            sources = [
                SourceItem(
                    source=doc.metadata.get("source", "Неизвестно"),
                    content=doc.page_content[:300],
                )
                for doc in docs
            ]

            return AskResponse(
                request_id=request_id,
                answer=answer,
                sources=sources,
                latency=round(latency, 3),
                model=request.model,
                model_version=model_version_str,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    if request.rating not in (0, 1):
        raise HTTPException(
            status_code=400, detail="rating должен быть 0 (дизлайк) или 1 (лайк)"
        )

    RAG_FEEDBACK_RATING_TOTAL.labels(rating=str(request.rating)).inc()

    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    file_exists = os.path.exists(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "request_id",
                "question",
                "answer",
                "rating",
                "comment",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request.request_id,
                "question": request.question,
                "answer": request.answer,
                "rating": request.rating,
                "comment": request.comment or "",
            }
        )

    with _optional_mlflow_run(run_name=f"Feedback_{request.request_id[:8]}") as _mf:
        if _mf:
            mlflow.log_param("request_id", request.request_id)
            mlflow.log_param("question", request.question)
            mlflow.log_metric("rating", request.rating)
            if request.comment:
                mlflow.log_text(request.comment, "feedback_comment.txt")

    return FeedbackResponse(status="ok", message="Feedback сохранён")


Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
