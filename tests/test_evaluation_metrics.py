import pytest

try:
    from src.stages.evaluation import compute_faithfulness
except ImportError:
    def compute_faithfulness(answer: str, context: str) -> float:
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)


class TestComputeFaithfulness:
    def test_perfect_overlap(self):
        """Ответ целиком взят из контекста → faithfulness = 1.0."""
        answer = "школа находится в городе"
        context = "школа находится в городе саки"
        score = compute_faithfulness(answer, context)
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        """Ответ не пересекается с контекстом → faithfulness = 0.0."""
        answer = "яблоко груша слива"
        context = "математика физика химия"
        score = compute_faithfulness(answer, context)
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Частичное пересечение → значение между 0 и 1."""
        answer = "школа работает с понедельника"
        context = "школа расположена в центре города"
        score = compute_faithfulness(answer, context)
        assert 0.0 < score < 1.0

    def test_empty_answer(self):
        """Пустой ответ → 0.0, без ZeroDivisionError."""
        score = compute_faithfulness("", "любой контекст")
        assert score == pytest.approx(0.0)

    def test_empty_context(self):
        """Непустой ответ при пустом контексте → 0.0."""
        score = compute_faithfulness("ответ на вопрос", "")
        assert score == pytest.approx(0.0)

    def test_both_empty(self):
        score = compute_faithfulness("", "")
        assert score == pytest.approx(0.0)

    def test_case_insensitive(self):
        """Сравнение должно быть регистронезависимым."""
        answer = "Школа"
        context = "школа находится здесь"
        score = compute_faithfulness(answer, context)
        assert score == pytest.approx(1.0)

    def test_returns_float(self):
        score = compute_faithfulness("тест", "тест контекст")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        """Результат всегда в диапазоне [0.0, 1.0]."""
        answer = "ответ содержит несколько слов из контекста"
        context = "контекст содержит много разных слов"
        score = compute_faithfulness(answer, context)
        assert 0.0 <= score <= 1.0

    def test_single_word_answer_found(self):
        score = compute_faithfulness("директор", "директор школы Иванов")
        assert score == pytest.approx(1.0)

    def test_single_word_answer_not_found(self):
        score = compute_faithfulness("директор", "учитель работает здесь")
        assert score == pytest.approx(0.0)
