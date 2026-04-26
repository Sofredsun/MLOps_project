import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class SimpleEmbedder:
    """
    Заглушка-эмбеддер (простая хеш-функция) для демо.
    Потом нужно заменить на HuggingFace.
    """

    def __init__(self):
        self.dim = 384  # Размерность как у multilingual-e5-small

    def embed_query(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.dim).astype(np.float32)


class MinimalDriftDetector:
    """MVP: Детектор дрейфа запросов"""

    def __init__(self, storage_path: str = "data/monitoring"):
        self.embedder = SimpleEmbedder()
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
        current-запросы — про совсем другие темы (дрейф гарантирован).
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
