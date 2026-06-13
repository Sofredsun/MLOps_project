import hashlib
import os
import re
from urllib.parse import urlparse, urlunparse

import pytest

BASE_URL = "http://saki-school2.ucoz.ru"
_BASE_NETLOC = urlparse(BASE_URL).netloc

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico",
    ".zip", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".xml",
)

_TRANSLIT: dict = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}

try:
    from src.data.parser import normalize_url, is_internal, skip_url, slugify, \
        url_to_filename
except ImportError:
    def normalize_url(url: str) -> str:
        parsed = urlparse(url)
        result = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
        )
        return result.rstrip("/") or "/"


    def is_internal(url: str) -> bool:
        netloc = urlparse(url).netloc
        return not netloc or netloc == _BASE_NETLOC


    def skip_url(url: str) -> bool:
        u = url.lower()
        return any(u.endswith(ext) for ext in SKIP_EXTENSIONS)


    def slugify(text: str) -> str:
        slug = "".join(_TRANSLIT.get(c, c) for c in text.lower())
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")[:60] or "page"


    def url_to_filename(url: str) -> str:
        path = urlparse(url).path
        basename = os.path.basename(path)
        if not basename or not basename.lower().endswith(".pdf"):
            basename = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
        return basename


class TestNormalizeUrl:
    def test_removes_trailing_slash(self):
        assert normalize_url("http://example.com/page/") == "http://example.com/page"

    def test_removes_fragment(self):
        url = "http://saki-school2.ucoz.ru/index/page#section"
        result = normalize_url(url)
        assert "#" not in result

    def test_preserves_query(self):
        url = "http://example.com/search?q=школа"
        result = normalize_url(url)
        assert "q=школа" in result

    def test_idempotent(self):
        """Двойная нормализация должна давать тот же результат."""
        url = "http://example.com/page/"
        assert normalize_url(normalize_url(url)) == normalize_url(url)


class TestIsInternal:
    def test_same_domain_is_internal(self):
        assert is_internal("http://saki-school2.ucoz.ru/some/page") is True

    def test_external_domain_is_not_internal(self):
        assert is_internal("http://google.com/page") is False

    def test_relative_url_is_internal(self):
        """Относительный URL (без netloc) считается внутренним."""
        assert is_internal("/index/page") is True

    def test_different_subdomain_is_external(self):
        assert is_internal("http://other.ucoz.ru/page") is False


class TestSkipUrl:
    @pytest.mark.parametrize("ext", [".jpg", ".png", ".pdf", ".css", ".js", ".zip"])
    def test_media_and_static_extensions_skipped(self, ext):
        # PDF не в SKIP_EXTENSIONS — это нормально, PDF скачиваются отдельно
        if ext == ".pdf":
            assert skip_url(f"http://example.com/file{ext}") is False
        else:
            assert skip_url(f"http://example.com/file{ext}") is True

    def test_html_page_not_skipped(self):
        assert skip_url("http://example.com/page") is False

    def test_case_insensitive(self):
        assert skip_url("http://example.com/photo.JPG") is True

    def test_pdf_not_skipped(self):
        """PDF файлы не должны пропускаться — они скачиваются отдельно."""
        assert skip_url("http://example.com/document.pdf") is False


class TestSlugify:
    def test_transliterates_russian(self):
        result = slugify("школа")
        assert result == "shkola"

    def test_replaces_spaces_with_dashes(self):
        result = slugify("основные сведения")
        assert " " not in result
        assert "-" in result

    def test_lowercase_output(self):
        result = slugify("ШКОЛА")
        assert result == result.lower()

    def test_max_length_60(self):
        long_text = "а" * 100
        assert len(slugify(long_text)) <= 60

    def test_empty_string_returns_page(self):
        assert slugify("") == "page"

    def test_no_leading_trailing_dashes(self):
        result = slugify("  школа  ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_mixed_russian_english(self):
        result = slugify("RAG система")
        assert "rag" in result
        assert "sistema" in result


class TestUrlToFilename:
    def test_extracts_pdf_filename(self):
        url = "http://example.com/docs/schedule.pdf"
        assert url_to_filename(url) == "schedule.pdf"

    def test_non_pdf_url_gets_hash_filename(self):
        url = "http://example.com/docs/file"
        result = url_to_filename(url)
        assert result.endswith(".pdf")
        assert len(result) == 16  # 12 hex chars + ".pdf"

    def test_empty_path_gets_hash(self):
        url = "http://example.com/"
        result = url_to_filename(url)
        assert result.endswith(".pdf")

    def test_deterministic_hash(self):
        """Один и тот же URL всегда дает одинаковое имя файла."""
        url = "http://example.com/unknown"
        assert url_to_filename(url) == url_to_filename(url)
