import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import mlflow
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel

from src.monitoring.drift_detector import MinimalDriftDetector

CHROMA_DIR = "chroma_langchain_db"
FEEDBACK_CSV = "data/models/feedback.csv"
AVAILABLE_MODELS = ["qwen2.5:7b", "llama3.2"]
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_K = 8

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

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("School_RAG_System")

app = FastAPI(
    title="School RAG API",
    description="API для RAG-системы школьного ИИ-ассистента",
    version="1.0.0"
)

_vector_store = None

drift_detector = MinimalDriftDetector()

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

    # Добавляем новый алерт в начало
    alerts.insert(0, result)
    # Храним только последние 50
    alerts = alerts[:50]

    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"}
        )
        _vector_store = Chroma(
            collection_name="school_knowledge_base",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    return _vector_store


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


class FeedbackRequest(BaseModel):
    request_id: str
    question: str
    answer: str
    rating: int  # 1 - лайк, 0 - дизлайк
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ЭНДПОИНТЫ
@app.get("/health")
def health():
    return {"status": "ok", "service": "School RAG API"}


@app.get("/monitoring/drift")
def check_drift(hours: int = 24):
    """
    Проверка дрейфа запросов
    ?hours=24 - анализировать последние 24 часа
    """
    result = drift_detector.detect_drift(hours=hours)

    # Если дрейф обнаружен, то логируем
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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{request.model}' недоступна. Доступные: {AVAILABLE_MODELS}"
        )

    request_id = str(uuid.uuid4())

    with mlflow.start_run(run_name=f"API_Query_{time.strftime('%H%M%S')}"):
        try:
            start_time = time.time()

            mlflow.log_param("request_id", request_id)
            mlflow.log_param("model", request.model)
            mlflow.log_param("k_retrieval", request.k_retrieval)
            mlflow.log_param("question", request.question)
            mlflow.log_param("embedding_model", "multilingual-e5-small")

            vector_store = get_vector_store()
            model = OllamaLLM(model=request.model, temperature=0.1)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": request.k_retrieval}
            )

            prompt = ChatPromptTemplate.from_template(TEMPLATE)
            chain = prompt | model

            docs = retriever.invoke(request.question)
            context_text = "\n\n".join([
                f"[Источник: {d.metadata.get('source', 'Неизвестно')}]"
                f"\n{d.page_content}"
                for d in docs
            ])

            answer = chain.invoke(
                {"context": context_text, "question": request.question})
            latency = time.time() - start_time

            mlflow.log_metric("latency", latency)
            mlflow.log_metric("context_length", len(context_text))
            mlflow.log_text(answer, "assistant_response.txt")

            sources = [
                SourceItem(
                    source=doc.metadata.get("source", "Неизвестно"),
                    content=doc.page_content[:300]
                )
                for doc in docs
            ]
            # Логирование запроса для мониторинга дрейфа
            try:
                drift_detector.log_query(
                    query=request.question,
                    request_id=request_id,
                    metadata={
                        "model": request.model,
                        "k": request.k_retrieval,
                        "latency": round(latency, 3),
                        "context_sources": len(docs)
                    }
                )
            except Exception as drift_error:
                print(f"Drift logging failed: {drift_error}")

            return AskResponse(
                request_id=request_id,
                answer=answer,
                sources=sources,
                latency=round(latency, 3),
                model=request.model
            )

        except Exception as e:
            # Логируем даже при ошибке основного запроса
            try:
                drift_detector.log_query(
                    query=request.question,
                    request_id=request_id,
                    metadata={"model": request.model, "error": str(e)[:100]}
                )
            except (OSError, RuntimeError, ValueError, TypeError):
                pass  # Безопасно игнорируем ошибки вторичного логирования

            raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    if request.rating not in (0, 1):
        raise HTTPException(status_code=400,
                            detail="rating должен быть 0 (дизлайк) или 1 (лайк)")

    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    file_exists = os.path.exists(FEEDBACK_CSV)

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "request_id", "question", "answer", "rating", "comment"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request.request_id,
            "question": request.question,
            "answer": request.answer,
            "rating": request.rating,
            "comment": request.comment or ""
        })

    # Логируем в MLflow тоже
    with mlflow.start_run(run_name=f"Feedback_{request.request_id[:8]}"):
        mlflow.log_param("request_id", request.request_id)
        mlflow.log_param("question", request.question)
        mlflow.log_metric("rating", request.rating)
        if request.comment:
            mlflow.log_text(request.comment, "feedback_comment.txt")

    return FeedbackResponse(status="ok", message="Feedback сохранён")
