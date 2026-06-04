import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _fake_embedding(seed: int = 0) -> list:
    """Детерминированный вектор размерностью 384 для воспроизводимости."""
    rng = np.random.default_rng(seed)
    return rng.random(384).tolist()


@pytest.fixture
def mock_embedder():
    """Mock-эмбеддер: embed_query возвращает разные векторы по seed из текста."""
    emb = MagicMock()
    emb.embed_query.side_effect = lambda text: _fake_embedding(seed=hash(text) % 1000)
    return emb


@pytest.fixture
def detector(tmp_path, mock_embedder):
    """MinimalDriftDetector с mock-эмбеддером и временной директорией хранилища."""
    with patch(
        "src.monitoring.drift_detector.HuggingFaceEmbedder", return_value=mock_embedder
    ):
        from src.monitoring.drift_detector import MinimalDriftDetector

        d = MinimalDriftDetector(storage_path=str(tmp_path))
    return d


@pytest.fixture
def concept_detector(tmp_path):
    """ConceptDriftDetector с временными путями к файлам."""
    from src.monitoring.drift_detector import ConceptDriftDetector

    d = ConceptDriftDetector(storage_path=str(tmp_path))
    # Перенаправляем feedback CSV во временную папку
    d.feedback_csv = tmp_path / "feedback.csv"
    return d


def _write_jsonl_records(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_feedback_csv(path: Path, rows: list[dict]):
    fieldnames = ["timestamp", "request_id", "question", "answer", "rating", "comment"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# MinimalDriftDetector — логирование запросов
class TestLogQuery:

    def test_creates_jsonl_file(self, detector, tmp_path):
        """log_query создает JSONL-файл если его не было."""
        assert not detector.queries_file.exists()
        detector.log_query("Когда каникулы?", request_id="r-001")
        assert detector.queries_file.exists()

    def test_record_has_required_fields(self, detector):
        """Каждая запись содержит все обязательные поля."""
        detector.log_query("Какое расписание?", request_id="r-002")
        record = json.loads(detector.queries_file.read_text(encoding="utf-8"))
        for field in ("request_id", "timestamp", "query", "embedding", "metadata"):
            assert field in record, f"Поле '{field}' отсутствует в записи"

    def test_embedding_stored_as_list(self, detector):
        """embedding сохраняется как список чисел, а не numpy-массив."""
        detector.log_query("test", request_id="r-003")
        record = json.loads(detector.queries_file.read_text())
        assert isinstance(record["embedding"], list)
        assert all(isinstance(v, float) for v in record["embedding"])

    def test_multiple_calls_append_records(self, detector):
        """Повторные вызовы дописывают строки, а не перезаписывают файл."""
        for i in range(5):
            detector.log_query(f"запрос {i}", request_id=f"r-{i}")
        lines = detector.queries_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_metadata_persisted(self, detector):
        """Произвольные метаданные сохраняются без потерь."""
        detector.log_query(
            "q", request_id="r-meta", metadata={"user": "alice", "lang": "ru"}
        )
        record = json.loads(detector.queries_file.read_text())
        assert record["metadata"] == {"user": "alice", "lang": "ru"}

    def test_empty_metadata_defaults_to_empty_dict(self, detector):
        detector.log_query("q", request_id="r-nometa")
        record = json.loads(detector.queries_file.read_text())
        assert record["metadata"] == {}


# MinimalDriftDetector — загрузка запросов
class TestLoadQueries:

    def test_returns_empty_list_when_no_file(self, detector):
        assert detector._load_queries() == []

    def test_loads_all_records_without_filter(self, detector):
        records = [
            {
                "request_id": f"r{i}",
                "timestamp": datetime.now().isoformat(),
                "query": "q",
                "embedding": _fake_embedding(i),
                "metadata": {},
            }
            for i in range(4)
        ]
        _write_jsonl_records(detector.queries_file, records)
        loaded = detector._load_queries()
        assert len(loaded) == 4

    def test_since_filter_excludes_old_records(self, detector):
        now = datetime.now()
        records = [
            {
                "request_id": "old",
                "timestamp": (now - timedelta(hours=48)).isoformat(),
                "query": "old",
                "embedding": _fake_embedding(0),
                "metadata": {},
            },
            {
                "request_id": "new",
                "timestamp": (now - timedelta(minutes=10)).isoformat(),
                "query": "new",
                "embedding": _fake_embedding(1),
                "metadata": {},
            },
        ]
        _write_jsonl_records(detector.queries_file, records)
        loaded = detector._load_queries(since=now - timedelta(hours=1))
        assert len(loaded) == 1
        assert loaded[0]["request_id"] == "new"

    def test_since_none_loads_everything(self, detector):
        records = [
            {
                "request_id": f"r{i}",
                "timestamp": (datetime.now() - timedelta(days=i * 10)).isoformat(),
                "query": "q",
                "embedding": _fake_embedding(i),
                "metadata": {},
            }
            for i in range(3)
        ]
        _write_jsonl_records(detector.queries_file, records)
        loaded = detector._load_queries(since=None)
        assert len(loaded) == 3


# MinimalDriftDetector — обнаружение дрейфа
class TestDetectDriftTestMode:

    def test_insufficient_data_fewer_than_10(self, detector):
        """При меньше 10 записях возвращается insufficient_data."""
        for i in range(7):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert result["status"] == "insufficient_data"

    def test_exactly_9_records_triggers_insufficient(self, detector):
        for i in range(9):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        assert detector.detect_drift(test_mode=True)["status"] == "insufficient_data"

    def test_10_records_returns_drift_result(self, detector):
        for i in range(10):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert "status" not in result  # нет поля insufficient_data
        assert "drift_score" in result

    def test_result_contains_all_required_keys(self, detector):
        for i in range(20):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        expected_keys = {
            "timestamp",
            "window_hours",
            "reference_size",
            "current_size",
            "drift_score",
            "centroid_shift",
            "norm_shift",
            "threshold",
            "drift_detected",
            "recommendation",
        }
        assert expected_keys.issubset(result.keys())

    def test_drift_detected_is_python_bool(self, detector):
        """drift_detected должен быть именно bool, а не np.bool_."""
        for i in range(20):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert type(result["drift_detected"]) is bool  # noqa: E721

    def test_drift_detected_consistent_with_score_and_threshold(self, detector):
        """drift_detected == True тогда и только тогда, когда score > threshold."""
        for i in range(20):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert result["drift_detected"] == (result["drift_score"] > result["threshold"])

    def test_drift_score_in_valid_range(self, detector):
        for i in range(20):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_sizes_sum_to_total_records(self, detector):
        n = 20
        for i in range(n):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=True)
        assert result["reference_size"] + result["current_size"] == n

    def test_no_file_returns_insufficient(self, detector):
        """detect_drift без сохраненных данных не падает, возвращает insufficient."""
        result = detector.detect_drift(test_mode=True)
        assert result["status"] == "insufficient_data"


class TestDetectDriftNormalMode:

    def test_fewer_than_50_total_returns_insufficient(self, detector):
        for i in range(30):
            detector.log_query(f"q{i}", request_id=f"r{i}")
        result = detector.detect_drift(test_mode=False)
        assert result["status"] == "insufficient_data"

    def test_no_data_returns_insufficient(self, detector):
        result = detector.detect_drift(test_mode=False)
        assert result["status"] == "insufficient_data"


# ConceptDriftDetector — логирование RAGAS-метрик
class TestLogQualityScore:

    def test_creates_ragas_jsonl_file(self, concept_detector, tmp_path):
        concept_detector.log_quality_score("req-1", "Вопрос?", 0.85, 0.92)
        assert concept_detector.ragas_file.exists()

    def test_record_values_are_stored_correctly(self, concept_detector):
        concept_detector.log_quality_score("req-1", "Вопрос?", 0.85, 0.92)
        record = json.loads(concept_detector.ragas_file.read_text())
        assert record["request_id"] == "req-1"
        assert record["faithfulness"] == pytest.approx(0.85, abs=1e-3)
        assert record["answer_relevancy"] == pytest.approx(0.92, abs=1e-3)

    def test_multiple_scores_appended(self, concept_detector):
        for i in range(4):
            concept_detector.log_quality_score(f"r{i}", "q", 0.7, 0.8)
        lines = concept_detector.ragas_file.read_text().strip().split("\n")
        assert len(lines) == 4


# ConceptDriftDetector — загрузка feedback
class TestLoadFeedback:

    def test_returns_empty_when_no_file(self, concept_detector):
        assert concept_detector._load_feedback(hours=24) == []

    def test_filters_out_old_records(self, concept_detector):
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": "old",
                "question": "q",
                "answer": "a",
                "rating": "1",
                "comment": "",
            },
            {
                "timestamp": (now - timedelta(minutes=30)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "request_id": "new",
                "question": "q",
                "answer": "a",
                "rating": "0",
                "comment": "",
            },
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        loaded = concept_detector._load_feedback(hours=1)
        assert len(loaded) == 1
        assert loaded[0]["request_id"] == "new"

    def test_loads_all_recent_records(self, concept_detector):
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(minutes=i * 5)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "request_id": f"r{i}",
                "question": "q",
                "answer": "a",
                "rating": "1",
                "comment": "",
            }
            for i in range(6)
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        loaded = concept_detector._load_feedback(hours=2)
        assert len(loaded) == 6


# ConceptDriftDetector — обнаружение concept drift
class TestDetectConceptDrift:

    def test_no_data_returns_stable(self, concept_detector):
        """Без данных дрейф не детектируется, ошибок нет."""
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is False
        assert result["issues"] == []

    def test_result_has_required_keys(self, concept_detector):
        result = concept_detector.detect_concept_drift(hours=24)
        for key in (
            "timestamp",
            "window_hours",
            "concept_drift_detected",
            "issues",
            "metrics",
            "recommendation",
        ):
            assert key in result

    def test_high_dislike_rate_triggers_drift(self, concept_detector):
        """Если ≥40% дизлайков и достаточно отзывов — дрейф обнаружен."""
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": f"r{i}",
                "question": "q",
                "answer": "a",
                "rating": "0",  # все дизлайки
                "comment": "",
            }
            for i in range(8)
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is True
        assert any("дизлайков" in issue for issue in result["issues"])

    def test_all_likes_no_drift(self, concept_detector):
        """100% лайков → дрейф не обнаружен."""
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": f"r{i}",
                "question": "q",
                "answer": "a",
                "rating": "1",
                "comment": "",
            }
            for i in range(8)
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is False

    def test_low_faithfulness_triggers_drift(self, concept_detector):
        """Faithfulness ниже порога (0.5) → дрейф обнаружен."""
        for i in range(5):
            concept_detector.log_quality_score(
                f"r{i}", "q", faithfulness=0.1, answer_relevancy=0.9
            )
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is True
        assert any("Faithfulness" in issue for issue in result["issues"])

    def test_low_relevancy_triggers_drift(self, concept_detector):
        """Answer Relevancy ниже порога (0.5) → дрейф обнаружен."""
        for i in range(5):
            concept_detector.log_quality_score(
                f"r{i}", "q", faithfulness=0.9, answer_relevancy=0.1
            )
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is True
        assert any("Relevancy" in issue for issue in result["issues"])

    def test_good_ragas_scores_no_drift(self, concept_detector):
        """Высокие RAGAS-метрики → нет дрейфа."""
        for i in range(5):
            concept_detector.log_quality_score(
                f"r{i}", "q", faithfulness=0.9, answer_relevancy=0.85
            )
        result = concept_detector.detect_concept_drift(hours=24)
        assert result["concept_drift_detected"] is False

    def test_metrics_dislike_rate_in_result(self, concept_detector):
        """dislike_rate присутствует в metrics при достаточном числе отзывов."""
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": f"r{i}",
                "question": "q",
                "answer": "a",
                "rating": str(i % 2),
                "comment": "",
            }
            for i in range(8)
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        result = concept_detector.detect_concept_drift(hours=24)
        assert "dislike_rate" in result["metrics"]

    def test_insufficient_feedback_noted_in_metrics(self, concept_detector):
        """Меньше min_feedback_count отзывов → заметка в metrics."""
        now = datetime.now()
        rows = [
            {
                "timestamp": (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": f"r{i}",
                "question": "q",
                "answer": "a",
                "rating": "0",
                "comment": "",
            }
            for i in range(2)  # < min_feedback_count = 5
        ]
        _write_feedback_csv(concept_detector.feedback_csv, rows)
        result = concept_detector.detect_concept_drift(hours=24)
        assert "feedback_note" in result["metrics"]

    def test_concept_drift_detected_is_bool(self, concept_detector):
        result = concept_detector.detect_concept_drift(hours=24)
        assert type(result["concept_drift_detected"]) is bool  # noqa: E721
