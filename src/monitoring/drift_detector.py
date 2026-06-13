import csv
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class HuggingFaceEmbedder:
    """Настоящий эмбеддер на базе multilingual-e5-small"""

    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        self._model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"},
        )
        self.dim = 384

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.embed_query(text)
        return np.array(vector, dtype=np.float32)


class MinimalDriftDetector:
    """MVP: Детектор дрейфа запросов"""

    def __init__(self, storage_path: str = "data/monitoring"):
        self.embedder = HuggingFaceEmbedder()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.queries_file = self.storage_path / "queries.jsonl"
        self.threshold = 0.15  # Порог дрейфа

    def log_query(self, query: str, request_id: str, metadata: Optional[Dict] = None):
        """Логирование запроса - вызывается в эндпоинте /ask"""
        embedding = self.embedder.embed_query(query)

        record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
        }

        # Добавляем в JSONL
        with open(self.queries_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_queries(self, since: datetime = None) -> List[Dict]:
        """Загрузка запросов из хранилища"""
        if not self.queries_file.exists():
            return []

        queries = []
        with open(self.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if (
                    since is None
                    or datetime.fromisoformat(record["timestamp"]) >= since
                ):
                    queries.append(record)
        return queries

    def detect_drift(self, hours: int = 24, test_mode: bool = False) -> Dict:
        """Проверка дрейфа - вызывается по расписанию или вручную"""
        now = datetime.now()

        all_queries = self._load_queries()

        if test_mode:
            # Тестовый режим: делим имеющиеся данные пополам
            # независимо от временных меток
            if len(all_queries) < 10:
                return {
                    "status": "insufficient_data",
                    "message": f"Нужно минимум 10 запросов для теста, сейчас: "
                    f"{len(all_queries)}",
                }
            mid = len(all_queries) // 2
            reference = all_queries[:mid]
            current = all_queries[mid:]
        else:
            reference_cutoff = now - timedelta(days=7)
            current_cutoff = now - timedelta(hours=hours)

            reference = [
                q
                for q in all_queries
                if datetime.fromisoformat(q["timestamp"]) < reference_cutoff
            ]
            current = [
                q
                for q in all_queries
                if datetime.fromisoformat(q["timestamp"]) >= current_cutoff
            ]

            if len(all_queries) < 50:
                return {
                    "status": "insufficient_data",
                    "message": "Нужно минимум 50 запросов",
                }

            if len(reference) < 20 or len(current) < 10:
                return {
                    "status": "insufficient_data",
                    "reference": len(reference),
                    "current": len(current),
                }

        # Извлекаем эмбеддинги
        ref_emb = np.array([q["embedding"] for q in reference])
        curr_emb = np.array([q["embedding"] for q in current])

        # Метрика 1: Сдвиг центроидов (косинусное расстояние)
        ref_centroid = np.mean(ref_emb, axis=0)
        curr_centroid = np.mean(curr_emb, axis=0)

        cos_sim = np.dot(ref_centroid, curr_centroid) / (
            np.linalg.norm(ref_centroid) * np.linalg.norm(curr_centroid) + 1e-8
        )
        centroid_shift = 1 - cos_sim

        # Метрика 2: Изменение распределения норм эмбеддингов (простой KS-like тест)
        ref_norms = np.linalg.norm(ref_emb, axis=1)
        curr_norms = np.linalg.norm(curr_emb, axis=1)

        # Простая эвристика: сдвиг среднего > 1.5σ
        norm_shift = abs(np.mean(curr_norms) - np.mean(ref_norms)) / (
            np.std(ref_norms) + 1e-8
        )

        # Агрегируем
        drift_score = max(centroid_shift, min(norm_shift / 3, 1.0))  # Нормализуем
        drift_detected = bool(drift_score > self.threshold)

        return {
            "timestamp": now.isoformat(),
            "window_hours": hours,
            "reference_size": len(reference),
            "current_size": len(current),
            "drift_score": round(float(drift_score), 3),
            "centroid_shift": round(float(centroid_shift), 3),
            "norm_shift": round(float(norm_shift), 2),
            "threshold": float(self.threshold),
            "drift_detected": bool(drift_detected),
            "recommendation": (
                "Проверьте базу знаний" if drift_detected else "Все стабильно"
            ),
        }

    def seed_test_data(self, n_reference: int = 30, n_current: int = 15):
        """
        Генерирует тестовые данные с искусственным дрейфом
        reference-запросы — про расписание/уроки,
        current-запросы — про совсем другие темы.
        """

        reference_queries = [
            "Во сколько начинается первый урок?",
            "Когда заканчиваются уроки в пятницу?",
            "Какое расписание на понедельник?",
            "Когда осенние каникулы?",
            "Сколько длится урок?",
            "Когда весенние каникулы?",
            "Во сколько начинается второй урок?",
            "Какое расписание звонков?",
        ]

        current_queries = [
            "Какая цена на обед в столовой?",
            "Что на завтрак сегодня?",
            "Есть ли вегетарианское меню?",
            "Сколько стоит питание в месяц?",
            "Как оплатить питание онлайн?",
        ]

        now = datetime.now()

        # Reference: 8+ дней назад
        for i in range(n_reference):
            query = reference_queries[i % len(reference_queries)] + f" (вариант {i})"
            embedding = self.embedder.embed_query(query)
            record = {
                "request_id": str(uuid.uuid4()),
                "timestamp": (now - timedelta(days=8, hours=i)).isoformat(),
                "query": query,
                "embedding": embedding.tolist(),
                "metadata": {"source": "test_seed"},
            }
            with open(self.queries_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Current: последние несколько часов
        for i in range(n_current):
            query = current_queries[i % len(current_queries)] + f" (вариант {i})"
            embedding = self.embedder.embed_query(query)
            record = {
                "request_id": str(uuid.uuid4()),
                "timestamp": (now - timedelta(minutes=i * 10)).isoformat(),
                "query": query,
                "embedding": embedding.tolist(),
                "metadata": {"source": "test_seed"},
            }
            with open(self.queries_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {"seeded_reference": n_reference, "seeded_current": n_current}


class ConceptDriftDetector:
    """Concept/Target Drift: падение качества генерации по feedback и RAGAS-метрикам"""

    def __init__(self, storage_path: str = "data/monitoring"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.feedback_csv = Path("data/models/feedback.csv")
        self.ragas_file = self.storage_path / "ragas_scores.jsonl"

        # Пороги
        self.dislike_rate_threshold = 0.4  # 40% дизлайков - алерт
        self.faithfulness_threshold = 0.5  # ниже 0.5 - алерт
        self.relevancy_threshold = 0.5  # ниже 0.5 - алерт
        self.min_feedback_count = 5  # минимум отзывов для анализа

    def log_quality_score(
        self,
        request_id: str,
        question: str,
        faithfulness: float,
        answer_relevancy: float,
    ):
        """Логирует метрики качества после каждого /ask"""
        record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(answer_relevancy, 3),
        }
        with open(self.ragas_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_feedback(self, hours: int = 24) -> List[Dict]:
        if not self.feedback_csv.exists():
            return []
        cutoff = datetime.now() - timedelta(hours=hours)
        rows = []
        with open(self.feedback_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        rows.append(row)
                except (ValueError, KeyError):
                    continue
        return rows

    def _load_ragas_scores(self, hours: int = 24) -> List[Dict]:
        if not self.ragas_file.exists():
            return []
        cutoff = datetime.now() - timedelta(hours=hours)
        scores = []
        with open(self.ragas_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if datetime.fromisoformat(record["timestamp"]) >= cutoff:
                    scores.append(record)
        return scores

    def detect_concept_drift(self, hours: int = 24) -> Dict:
        now = datetime.now()
        issues = []
        metrics = {}

        # Анализ feedback (дизлайки)
        feedback_rows = self._load_feedback(hours)
        if len(feedback_rows) >= self.min_feedback_count:
            total = len(feedback_rows)
            dislikes = sum(1 for r in feedback_rows if str(r.get("rating", "1")) == "0")
            dislike_rate = dislikes / total
            metrics["feedback_total"] = total
            metrics["dislike_count"] = dislikes
            metrics["dislike_rate"] = round(dislike_rate, 3)

            if dislike_rate >= self.dislike_rate_threshold:
                issues.append(
                    f"Высокий процент дизлайков: {dislike_rate:.0%} "
                    f"({dislikes}/{total}) за последние {hours}ч"
                )
        else:
            metrics["feedback_total"] = len(feedback_rows)
            metrics["feedback_note"] = (
                f"Недостаточно отзывов (нужно {self.min_feedback_count})"
            )

        # Анализ RAGAS-метрик
        ragas_scores = self._load_ragas_scores(hours)
        if ragas_scores:
            avg_faithfulness = np.mean([s["faithfulness"] for s in ragas_scores])
            avg_relevancy = np.mean([s["answer_relevancy"] for s in ragas_scores])
            metrics["ragas_samples"] = len(ragas_scores)
            metrics["avg_faithfulness"] = round(float(avg_faithfulness), 3)
            metrics["avg_answer_relevancy"] = round(float(avg_relevancy), 3)

            if avg_faithfulness < self.faithfulness_threshold:
                issues.append(
                    f"Низкий Faithfulness: {avg_faithfulness:.3f} "
                    f"(порог: {self.faithfulness_threshold})"
                )
            if avg_relevancy < self.relevancy_threshold:
                issues.append(
                    f"Низкий Answer Relevancy: {avg_relevancy:.3f} "
                    f"(порог: {self.relevancy_threshold})"
                )
        else:
            metrics["ragas_note"] = "Нет RAGAS-метрик за период"

        drift_detected = len(issues) > 0

        return {
            "timestamp": now.isoformat(),
            "window_hours": hours,
            "concept_drift_detected": drift_detected,
            "issues": issues,
            "metrics": metrics,
            "recommendation": (
                "Проверьте качество ответов и базу знаний"
                if drift_detected
                else "Качество генерации стабильно"
            ),
        }
