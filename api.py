import sys
from pathlib import Path

# Добавляем src в путь для импорта utils
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import csv
import io
import json
import os
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from src.monitoring.drift_detector import ConceptDriftDetector, MinimalDriftDetector
from utils.config import PATHS  # noqa: E402

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

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
_mlflow_exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "School_RAG_System")
try:
    mlflow.set_experiment(_mlflow_exp)
except Exception:
    pass

# Глобальный статус переобучения
_retrain_status = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "",
}


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
        pass


RAG_FEEDBACK_RATING_TOTAL = Counter(
    "rag_feedback_rating_total",
    "Отзывы пользователей по оценке rating (0 — дизлайк, 1 — лайк).",
    ("rating",),
)

app = FastAPI(
    title="School RAG API",
    description="API для RAG-системы школьного ИИ-ассистента",
    version="1.0.0",
)
drift_detector = MinimalDriftDetector()
concept_detector = ConceptDriftDetector()
_vector_store = None

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
            model_name="intfloat/multilingual-e5-small", model_kwargs={"device": "cpu"}
        )
        _vector_store = Chroma(
            collection_name="school_knowledge_base",
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _vector_store


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


@app.get("/monitoring/report")
def get_drift_report():
    """
    Генерирует HTML-отчет о дрейфе и возвращает как скачиваемый файл.
    Вызывается кнопкой "Скачать отчет" в Streamlit.
    """
    # Берем последний сохраненный алерт вместо пересчета
    drift_result = {}
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                alerts = json.load(f)
                drift_result = alerts[0] if alerts else {}
        except (json.JSONDecodeError, IndexError):
            drift_result = {}

    # Если алертов нет — считаем заново
    if not drift_result:
        drift_result = drift_detector.detect_drift(hours=24)

    concept_result = {}
    if CONCEPT_ALERTS_FILE.exists():
        try:
            with open(CONCEPT_ALERTS_FILE, "r", encoding="utf-8") as f:
                concept_alerts = json.load(f)
                concept_result = concept_alerts[0] if concept_alerts else {}
        except (json.JSONDecodeError, IndexError):
            concept_result = {}

    if not concept_result:
        concept_result = concept_detector.detect_concept_drift(hours=24)

    # История алертов (data drift)
    drift_history = []
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE, "r", encoding="utf-8") as f:
                drift_history = json.load(f)
        except json.JSONDecodeError:
            drift_history = []

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Строим график drift score по истории
    chart_rows = ""
    if drift_history:
        max_score = (
            max((a.get("drift_score", 0) for a in drift_history), default=1) or 1
        )
        for alert in drift_history[-20:]:  # последние 20 записей
            score = alert.get("drift_score", 0)
            ts = alert.get("timestamp", "")[:16]
            bar_pct = int(score / max_score * 100)
            color = "#ef4444" if alert.get("drift_detected") else "#22c55e"
            chart_rows += f"""
            <tr>
              <td style="padding:4px 8px;font-size:12px;color:#94a3b8">{ts}</td>
              <td style="padding:4px 8px">
                <div style="background:{color};width:{bar_pct}%;min-width:4px;
                            height:16px;border-radius:3px"></div>
              </td>
              <td style="padding:4px 8px;font-size:12px;color:#e2e8f0">{score:.3f}</td>
            </tr>"""

    # Метрики concept drift
    metrics = concept_result.get("metrics", {})
    metrics_rows = ""
    metric_map = {
        "feedback_total": "Всего отзывов",
        "dislike_count": "Дизлайков",
        "dislike_rate": "Процент дизлайков",
        "avg_faithfulness": "Faithfulness (avg)",
        "avg_answer_relevancy": "Answer Relevancy (avg)",
        "ragas_samples": "RAGAS замеров",
    }
    for key, label in metric_map.items():
        if key in metrics:
            val = metrics[key]
            if key == "dislike_rate":
                val = f"{val:.0%}"
            status_color = ""
            if (
                key == "dislike_rate"
                and isinstance(metrics[key], float)
                and metrics[key] >= 0.4
            ):
                status_color = "color:#ef4444;font-weight:bold"
            elif (
                key in ("avg_faithfulness", "avg_answer_relevancy")
                and isinstance(val, float)
                and val < 0.5
            ):
                status_color = "color:#ef4444;font-weight:bold"
            metrics_rows += f"""
            <tr>
              <td style="padding:6px 12px;color:#94a3b8">{label}</td>
              <td style="padding:6px 12px;{status_color}">{val}</td>
            </tr>"""

    # Рекомендации
    recommendations = []
    if drift_result.get("drift_detected"):
        recommendations.append(
            "Обнаружен дрейф запросов — тема вопросов сильно изменился. "
            "Рекомендуется обновить базу знаний ChromaDB."
        )
    if concept_result.get("concept_drift_detected"):
        for issue in concept_result.get("issues", []):
            recommendations.append(f"{issue}")
    if not recommendations:
        recommendations.append("Дрейф не обнаружен. Система работает стабильно.")

    rec_html = "".join(
        f"<li style='margin:6px 0;color:#e2e8f0'>{r}</li>" for r in recommendations
    )

    # Data drift статус
    drift_score = drift_result.get("drift_score", "—")
    drift_status_color = "#ef4444" if drift_result.get("drift_detected") else "#22c55e"
    drift_status_text = (
        "ДРЕЙФ ОБНАРУЖЕН" if drift_result.get("drift_detected") else "СТАБИЛЬНО"
    )
    concept_status_color = (
        "#ef4444" if concept_result.get("concept_drift_detected") else "#22c55e"
    )
    concept_status_text = (
        "ДРЕЙФ ОБНАРУЖЕН"
        if concept_result.get("concept_drift_detected")
        else "СТАБИЛЬНО"
    )

    chart_html = (
        "<table>" + chart_rows + "</table>"
        if chart_rows
        else '<p style="color:#475569;font-size:13px">'
        "Нет исторических данных об алертах.</p>"
    )

    metrics_html = (
        "<table>" + metrics_rows + "</table>"
        if metrics_rows
        else '<p style="color:#475569;font-size:13px">'
        "Нет данных о метриках за последние 24 часа.</p>"
    )

    drift_score_style = (
        f"margin-top:12px;font-size:28px;font-weight:bold;color:{drift_status_color}"
    )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Отчёт о дрейфе — {now_str}</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#0f172a;
    color:#e2e8f0; margin:0; padding:32px }}
    h1 {{ color:#f8fafc; font-size:24px; margin-bottom:4px }}
    h2 {{ color:#94a3b8; font-size:14px; margin-bottom:32px; font-weight:normal }}
    h3 {{ color:#cbd5e1; font-size:16px; margin:24px 0 12px }}
    .card {{ background:#1e293b; border-radius:12px; padding:20px; margin-bottom:20px }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:20px;
              font-size:13px; font-weight:bold; color:#fff }}
    table {{ width:100%; border-collapse:collapse }}
    tr:hover td {{ background:#1e3a5f20 }}
    .footer {{ color:#475569; font-size:12px; margin-top:32px; text-align:center }}
  </style>
</head>
<body>
  <h1>Отчет о дрейфе данных</h1>
  <h2>Сгенерирован: {now_str} &nbsp;|&nbsp; Окно анализа: последние 24 часа</h2>

  <!-- Статус карточки -->
  <div style="display:flex;gap:16px;margin-bottom:20px">
    <div class="card" style="flex:1">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">Data Drift (запросы)
      </div>
      <span class="badge" style="background:{drift_status_color}">{drift_status_text}
      </span>
      <div style="{drift_score_style}">
        {drift_score}
      </div>
      <div style="font-size:12px;color:#64748b">drift score (порог:
{drift_result.get('threshold', 0.15)})</div>
    </div>
    <div class="card" style="flex:1">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">
      Concept Drift (качество)</div>
      <span class="badge" style="background:{concept_status_color}">
{concept_status_text}</span>
      <div style="margin-top:12px;font-size:13px;color:#94a3b8">
        {len(concept_result.get('issues', []))} проблем выявлено
      </div>
    </div>
    <div class="card" style="flex:1">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">Размер выборки</div>
      <div style="font-size:28px;font-weight:bold;color:#60a5fa">
        {drift_result.get('current_size', '—')}
      </div>
      <div style="font-size:12px;color:#64748b">запросов в текущем окне</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px">
        reference: {drift_result.get('reference_size', '—')}
      </div>
    </div>
  </div>

  <!-- График drift score -->
  <div class="card">
    <h3>История Drift Score</h3>
    {chart_html}
  </div>

  <!-- Метрики качества -->
  <div class="card">
    <h3>Метрики качества (Concept Drift)</h3>
    {metrics_html}
  </div>

  <!-- Рекомендации -->
  <div class="card">
    <h3>Рекомендации</h3>
    <ul style="margin:0;padding-left:20px">{rec_html}</ul>
  </div>

  <div class="footer">School RAG System &nbsp;|&nbsp; Отчёт сгенерирован автоматически
  </div>
</body>
</html>"""

    filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    return StreamingResponse(
        io.BytesIO(html.encode("utf-8")),
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/retrain/status")
def retrain_status():
    """Возвращает текущий статус переобучения"""
    return _retrain_status


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

    with _optional_mlflow_run(run_name=f"API_Query_{time.strftime('%H%M%S')}") as _mf:
        try:
            start_time = time.time()

            if _mf:
                mlflow.log_param("request_id", request_id)
                mlflow.log_param("model", request.model)
                mlflow.log_param("k_retrieval", request.k_retrieval)
                mlflow.log_param("question", request.question)
                mlflow.log_param("embedding_model", "multilingual-e5-small")

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
            if hasattr(answer, "content"):
                answer = answer.content
            answer = str(answer).strip()
            latency = time.time() - start_time

            if _mf:
                mlflow.log_metric("latency", latency)
                mlflow.log_metric("context_length", len(context_text))
                mlflow.log_text(answer, "assistant_response.txt")

            # Считаем RAGAS-метрики
            try:
                faith = _compute_faithfulness(answer, context_text)
                relevancy = _compute_answer_relevancy(request.question, answer)

                concept_detector.log_quality_score(
                    request_id=request_id,
                    question=request.question,
                    faithfulness=faith,
                    answer_relevancy=relevancy,
                )
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


@app.post("/retrain")
def retrain():
    """
    Запускает переиндексацию ChromaDB и логирует MLflow Run.
    Выполняется в фоновом потоке — не блокирует API.
    """

    if _retrain_status["status"] == "running":
        return {
            "status": "already_running",
            "message": "Обновление базы знаний уже запущено",
        }

    def _do_retrain():
        global _vector_store
        _retrain_status.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "message": "Переиндексация запущена...",
            }
        )
        try:
            with _optional_mlflow_run(run_name="Manual_Retrain") as active:

                # Пытаемся получить новые данные через DVC
                try:
                    result = subprocess.run(
                        ["dvc", "pull", "--force"],
                        cwd="/app",
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    dvc_status = "success" if result.returncode == 0 else "skipped"
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    dvc_status = "skipped"

                # Переиндексируем ChromaDB
                try:
                    for stage_script in [
                        "src/stages/download_data.py",
                        "src/stages/splitter.py",
                    ]:
                        subprocess.run(
                            [sys.executable, stage_script],
                            cwd="/app",
                            check=True,
                            timeout=300,
                            env={**os.environ, "PYTHONPATH": "/app/src"},
                        )
                    pipeline_status = "success"
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as e:
                    pipeline_status = f"failed: {str(e)}"

                # Логируем в MLflow
                if active:
                    mlflow.log_param("trigger", "manual_ui")
                    mlflow.log_param("dvc_pull_status", dvc_status)
                    mlflow.log_param("pipeline_status", pipeline_status)
                    mlflow.log_metric("retrain_triggered", 1)

            # Stage 3 только создание ChromaDB без полного тестирования
            from src.stages.evaluation import (
                create_chroma_db,
                initialize_embeddings,
                load_chunks,
            )

            embeddings = initialize_embeddings()
            train_chunks = load_chunks(PATHS.TRAIN_DIR, "training")
            create_chroma_db(train_chunks, embeddings)

            # Сбрасываем кеш
            _vector_store = None
            # Переинициализируем
            _vector_store = get_vector_store()

            # Пересчитываем дрейф
            new_drift = drift_detector.detect_drift(hours=24)
            new_concept = concept_detector.detect_concept_drift(hours=24)

            if not new_drift.get("drift_detected"):
                if ALERTS_FILE.exists():
                    ALERTS_FILE.write_text("[]", encoding="utf-8")

            if not new_concept.get("concept_drift_detected"):
                if CONCEPT_ALERTS_FILE.exists():
                    CONCEPT_ALERTS_FILE.write_text("[]", encoding="utf-8")

            _retrain_status.update(
                {
                    "status": "done",
                    "finished_at": datetime.now().isoformat(),
                    "message": (
                        f"Переиндексация завершена. "
                        f"DVC: {dvc_status}, Pipeline: {pipeline_status}"
                    ),
                }
            )

        except Exception as e:
            _retrain_status.update(
                {
                    "status": "error",
                    "finished_at": datetime.now().isoformat(),
                    "message": f"Ошибка: {str(e)}",
                }
            )

    thread = threading.Thread(target=_do_retrain, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Обновление базы знаний запущено в фоне",
    }


Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health", "/docs", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
