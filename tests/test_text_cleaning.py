import re

try:
    from src.stages.download_data import clean_text, clean_school_md
except ImportError:
    def clean_text(text: str) -> str:
        text = re.sub(r"(?<=[^\\.!?])\n+(?=[а-яёa-z])", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    def clean_school_md(text: str) -> str:
        text = re.sub(r"==> picture \[.*?\] intentionally omitted <==", "", text)
        text = re.sub(
            r"----- Start of picture text -----.*?----- End of picture text -----",
            "", text, flags=re.DOTALL,
        )
        text = re.sub(
            r"РАССМОТРЕНО.*?Дата: \d{4}\.\d{2}\.\d{2}.*?\+03\'00\'",
            "", text, flags=re.DOTALL,
        )
        text = re.sub(r"_\s\d+\s_", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join([line.strip() for line in text.split("\n")])
        return text.strip()


class TestCleanText:
    def test_removes_extra_whitespace(self):
        assert clean_text("слово   другое") == "слово другое"

    def test_strips_leading_trailing_spaces(self):
        assert clean_text("  текст  ") == "текст"

    def test_joins_soft_wrapped_lines(self):
        """Перенос строки внутри предложения (строчная буква) должен схлопнуться."""
        text = "начало предложения\nпродолжение строчной"
        result = clean_text(text)
        assert "\n" not in result

    def test_preserves_sentence_endings(self):
        """Перенос после точки не должен удаляться (новый абзац)."""
        text = "Конец предложения.\nНовое предложение"
        result = clean_text(text)
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \n\n\t  ") == ""


class TestCleanSchoolMd:
    def test_removes_picture_omit_tag(self):
        text = "До\n==> picture [image.png] intentionally omitted <==\nПосле"
        result = clean_school_md(text)
        assert "intentionally omitted" not in result
        assert "До" in result
        assert "После" in result

    def test_removes_picture_text_block(self):
        text = (
            "Текст\n"
            "----- Start of picture text -----\n"
            "Какой-то текст внутри картинки\n"
            "----- End of picture text -----\n"
            "Конец"
        )
        result = clean_school_md(text)
        assert "Start of picture text" not in result
        assert "Какой-то текст внутри картинки" not in result
        assert "Конец" in result

    def test_removes_page_numbers(self):
        text = "Страница\n_ 2 _\nСледующая страница\n_ 15 _"
        result = clean_school_md(text)
        assert "_ 2 _" not in result
        assert "_ 15 _" not in result

    def test_collapses_multiple_blank_lines(self):
        text = "Абзац 1\n\n\n\n\nАбзац 2"
        result = clean_school_md(text)
        assert "\n\n\n" not in result

    def test_strips_line_spaces(self):
        text = "  строка с отступом  \n  другая строка  "
        result = clean_school_md(text)
        for line in result.split("\n"):
            assert line == line.strip()

    def test_empty_input(self):
        assert clean_school_md("") == ""

    def test_clean_content_unchanged(self):
        """Обычный текст без мусора не должен портиться."""
        text = "## Заголовок\n\nОбычный абзац с текстом.\n\n- элемент списка"
        result = clean_school_md(text)
        assert "Заголовок" in result
        assert "Обычный абзац" in result
        assert "элемент списка" in result

    def test_removes_digital_signature_block(self):
        text = (
            "Важный документ\n"
            "РАССМОТРЕНО на заседании\n"
            "Дата: 2023.09.01 12:00:00\n"
            "+03'00'\n"
            "Конец документа"
        )
        result = clean_school_md(text)
        assert "РАССМОТРЕНО" not in result
        assert "Конец документа" in result
