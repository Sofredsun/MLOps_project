"""
Парсер сайта http://saki-school2.ucoz.ru/
Фиксированный список страниц + рекурсивный обход
Контент берется только из div.content
PDF-файлы скачиваются в school_knowledge_base/docs/
Сканы (без текстового слоя) пропускаются
"""

import hashlib
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_URL = "http://saki-school2.ucoz.ru"
OUTPUT_DIR = PROJECT_ROOT / "data" / "school_knowledge_base" / "pages"
DOCS_DIR = PROJECT_ROOT / "data" / "school_knowledge_base" / "docs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DELAY = 1.2
PDF_MIN_CHARS = 100

# Парсим netloc один раз
_BASE_NETLOC = urlparse(BASE_URL).netloc

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

TARGET_PAGES = [
    (
        "http://saki-school2.ucoz.ru/index/osnovnye_svedenija/0-58",
        "основные_сведения",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/struktura_i_organy_upravlenija_obrazovatelnoj_organizaciej/0-59",
        "структура_и_органы_управления",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/rukovodstvo_pedagogicheskij_sostav/0-14",
        "руководство_педсостав",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/pedagogicheskij_sostav/0-203",
        "педагогический_состав",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/materialno_tekhnicheskoe_obespechenie_i_osnashhennost_obrazovatelnogo_processa/0-56",
        "материально_техническое",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/organizacija_pitanija_v_obrazovatelnoj_organizacii/0-173",
        "питание",
        True,
    ),
    ("http://saki-school2.ucoz.ru/index/dokumentacija/0-42", "документация", True),
    (
        "http://saki-school2.ucoz.ru/index/raspisanie_urokov/0-50",
        "расписание_уроков",
        True,
    ),
    (
        "http://saki-school2.ucoz.ru/index/rezhim_raboty_shkoly/0-51",
        "режим_работы",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/vserossijskie_proverochnye_raboty/0-76",
        "всероссийские_проверочные",
        False,
    ),
    (
        "http://saki-school2.ucoz.ru/index/vneurochnaja_dejatelnost/0-55",
        "внеурочная_деятельность",
        True,
    ),
    ("http://saki-school2.ucoz.ru/publ/", "публикации", True),
]

SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".svg",
    ".ico",
    ".zip",
    ".rar",
    ".7z",
    ".mp3",
    ".mp4",
    ".avi",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ttf",
    ".xml",
)

# Словарь транслитерации
_TRANSLIT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

# Паттерн «мусорных» слов
_PDF_NOISE = re.compile(
    r"\s*(скачать\.{0,3}|посмотреть\.{0,3}|открыть карточку сотрудника\.{0,3}"
    r"|открыть\.{0,3}|просмотр\.{0,3}|view\.{0,3}|download\.{0,3})\s*",
    re.IGNORECASE,
)

# Паттерн для удаления "(открыть карточку сотрудника)"
_STAFF_CARD_NOISE = re.compile(
    r"\(\s*открыть карточку сотрудника\s*\)|\(\s*\)", re.IGNORECASE
)

# Метки страниц, для которых .md не создается — только скачиваются PDF
_PDF_ONLY_LABELS = {"документация", "питание"}

# Заголовки блоков в ВПР, после которых контент пропускается
_VPR_SKIP_HEADINGS = {"документация", "дополнительные материалы"}

# Метка страницы педсостава
_STAFF_LABEL = "педагогический_состав"


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


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slugify(text: str) -> str:
    slug = "".join(_TRANSLIT.get(c, c) for c in text.lower())
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60] or "page"


def unique_filename(base: str, used: set) -> str:
    name, i = base, 2
    while name in used:
        name = f"{base}-{i}"
        i += 1
    used.add(name)
    return name


def url_to_filename(url: str) -> str:
    """Превращает URL PDF в имя файла, сохраняя оригинальное имя."""
    path = urlparse(url).path
    basename = os.path.basename(path)
    # если имя пустое или конфликтное — берем хэш
    if not basename or not basename.lower().endswith(".pdf"):
        basename = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
    return basename


def _clean_pdf_noise(text: str) -> str:
    """Убирает «Скачать...», «Посмотреть...» и подобное из строки текста."""
    cleaned = _PDF_NOISE.sub(" ", text).strip()
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# HTTP


def get_page(url: str, session: requests.Session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code != 200:
            print(f"    [HTTP {resp.status_code}] {url}")
            return None
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None
        return BeautifulSoup(resp.text, "lxml")
    except requests.exceptions.TooManyRedirects:
        print(f"    [redirect loop] {url}")
        return None
    except requests.RequestException as e:
        print(f"    [{type(e).__name__}] {url}")
        return None


# PDF скачивание и проверка


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Извлекает текст из байтов PDF через pdfminer. Возвращает пустую строку при ошибке."""
    try:
        import io

        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        out = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            out,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        return out.getvalue()
    except (
        Exception
    ):  # noqa: BLE001 — pdfminer бросает самые разные внутренние исключения
        return ""


def is_document_pdf(pdf_bytes: bytes) -> bool:
    """True, если PDF содержит достаточно текста (не скан)."""
    text = extract_pdf_text(pdf_bytes)
    return len(text.strip()) >= PDF_MIN_CHARS


def _pdf_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_pdf(url: str, session: requests.Session, downloaded: dict) -> str | None:
    """
    Скачивает PDF, проверяет на скан.
    Возвращает локальный путь к файлу или None (если скан / ошибка).
    """
    if url in downloaded:
        return downloaded[url]

    try:
        resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            print(f"    [PDF HTTP {resp.status_code}] {url}")
            downloaded[url] = None
            return None
        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
            downloaded[url] = None
            return None

        pdf_bytes = resp.content
        if not is_document_pdf(pdf_bytes):
            print(f"    [скан, пропущен] {url}")
            downloaded[url] = None
            return None

        filename = url_to_filename(url)
        # избегаем коллизий имен
        base, ext = os.path.splitext(filename)
        candidate = filename
        counter = 2

        # Сравниваем файлы по MD5-хэшу
        new_hash = _pdf_hash(pdf_bytes)
        while os.path.exists(os.path.join(DOCS_DIR, candidate)):
            existing_path = os.path.join(DOCS_DIR, candidate)
            with open(existing_path, "rb") as ef:
                if _pdf_hash(ef.read()) == new_hash:
                    downloaded[url] = existing_path
                    return existing_path
            candidate = f"{base}_{counter}{ext}"
            counter += 1

        local_path = os.path.join(DOCS_DIR, candidate)
        with open(local_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"    [PDF сохранен] {candidate}")
        downloaded[url] = local_path
        return local_path

    except requests.RequestException as e:
        print(f"    [PDF ошибка: {type(e).__name__}] {url}")
        downloaded[url] = None
        return None


# Парсинг


def parse_table_md(table_tag, cell_clean=None) -> str:
    def _cell(td):
        txt = clean(td.get_text())
        return cell_clean(txt) if cell_clean else txt

    headers, rows = [], []
    thead = table_tag.find("thead")
    if thead:
        headers = [_cell(c) for c in thead.find_all(["th", "td"])]

    header_kw = {
        "фио",
        "имя",
        "предмет",
        "класс",
        "пн",
        "вт",
        "ср",
        "чт",
        "пт",
        "урок",
        "время",
        "учитель",
        "должность",
        "№",
        "кабинет",
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
    }
    first = True
    tbody = table_tag.find("tbody") or table_tag
    for tr in tbody.find_all("tr"):
        cells = [_cell(td) for td in tr.find_all(["td", "th"])]
        if not any(cells):
            continue
        if first and not headers and any(c.lower() in header_kw for c in cells):
            headers = cells
            first = False
            continue
        rows.append(cells)
        first = False

    if not rows:
        return ""

    col = max(len(headers), max((len(r) for r in rows), default=0))
    lines = []
    if headers:
        lines.append("| " + " | ".join((headers + [""] * col)[:col]) + " |")
        lines.append("|" + " --- |" * col)
    else:
        lines.append("|" + " --- |" * col)
    for row in rows:
        lines.append("| " + " | ".join((row + [""] * col)[:col]) + " |")
    return "\n".join(lines)


def collect_pdf_links(content_el, base_url: str) -> list[dict]:
    """
    Собирает все PDF-ссылки из контента (дедупликация по URL).
    Возвращает список {'text': str, 'url': str}.
    Ссылки вида «Скачать» без «Посмотреть» пропускаются — они дублируют тот же файл.
    """
    seen_urls: set = set()
    pdfs: list = []
    for a in content_el.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        link_text = clean(a.get_text()) or "документ"
        link_lower = link_text.lower()

        is_pdf_url = href.lower().endswith(".pdf")
        is_view = any(
            w in link_lower for w in ("посмотреть", "открыть", "просмотр", "view")
        )
        is_download = any(w in link_lower for w in ("скачать", "download", "загрузить"))

        if is_download and not is_view:
            continue

        if (is_pdf_url or is_view) and full not in seen_urls:
            seen_urls.add(full)
            pdfs.append({"text": link_text, "url": full})
    return pdfs


def content_to_markdown(
    content_el,
    page_url: str,
    session: requests.Session,
    downloaded: dict,
    label: str = "",
) -> str:
    """
    Конвертирует div.content в Markdown.
    """
    md_lines = []
    seen_texts: set = set()

    # Флаг: находимся внутри пропускаемого блока (для ВПР)
    is_vpr = label == "всероссийские_проверочные"
    skip_block = False  # True — текущий блок пропускается

    def add(line: str):
        if not line:
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            return
        if line not in seen_texts:
            md_lines.append(line)
            seen_texts.add(line)

    # Множество URL, которые уже обработаны — чтобы не скачивать дважды
    handled_pdf_urls: set = set()

    def handle_pdf_links(links: list):
        """Скачивает PDF-ссылки, не добавляя их в markdown."""
        for lnk in links:
            if lnk["url"] not in handled_pdf_urls:
                handled_pdf_urls.add(lnk["url"])
                download_pdf(lnk["url"], session, downloaded)

    def process_node(el, depth=0):
        nonlocal skip_block
        if not isinstance(el, Tag):
            return

        # Таблица
        if el.name == "table":
            if skip_block:
                return
            cell_clean = (
                (lambda t: _STAFF_CARD_NOISE.sub("", t).strip())
                if label == _STAFF_LABEL
                else None
            )
            tbl = parse_table_md(el, cell_clean=cell_clean)
            if tbl:
                add("")
                for line in tbl.splitlines():
                    md_lines.append(line)
                add("")
            return

        # Заголовки
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            txt = clean(el.get_text())
            if not txt:
                return
            # ВПР: проверяем, не начинается ли пропускаемый блок
            if is_vpr and txt.lower() in _VPR_SKIP_HEADINGS:
                skip_block = True
                return
            # Любой другой заголовок — выходим из режима пропуска
            if is_vpr:
                skip_block = False
            add("")
            add("#" * level + " " + txt)
            add("")
            return

        # Параграф
        if el.name == "p":
            # ВПР: проверяем текст параграфа как псевдо-заголовок блока
            raw_txt = clean(el.get_text())
            if is_vpr and raw_txt.rstrip(":").lower() in _VPR_SKIP_HEADINGS:
                skip_block = True
                return
            if skip_block:
                # PDF внутри пропускаемого блока всё равно скачиваем
                handle_pdf_links(collect_pdf_links(el, page_url))
                return
            links_in_p = collect_pdf_links(el, page_url)
            if links_in_p:
                txt = _clean_pdf_noise(raw_txt)
                # Педсостав: убираем "(открыть карточку сотрудника)"
                if label == _STAFF_LABEL:
                    txt = _STAFF_CARD_NOISE.sub("", txt).strip()
                if txt:
                    add(txt)
                    add("")
                handle_pdf_links(links_in_p)
            else:
                txt = raw_txt
                if label == _STAFF_LABEL:
                    txt = _STAFF_CARD_NOISE.sub("", txt).strip()
                if txt:
                    add(txt)
                    add("")
            return

        # Ненумерованный список
        if el.name == "ul":
            if skip_block:
                for li in el.find_all("li", recursive=False):
                    handle_pdf_links(collect_pdf_links(li, page_url))
                return
            for li in el.find_all("li", recursive=False):
                links_in_li = collect_pdf_links(li, page_url)
                txt = _clean_pdf_noise(clean(li.get_text()))
                if label == _STAFF_LABEL:
                    txt = _STAFF_CARD_NOISE.sub("", txt).strip()
                if txt:
                    add(f"- {txt}")
                if links_in_li:
                    handle_pdf_links(links_in_li)
            add("")
            return

        # Нумерованный список
        if el.name == "ol":
            if skip_block:
                for li in el.find_all("li", recursive=False):
                    handle_pdf_links(collect_pdf_links(li, page_url))
                return
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                links_in_li = collect_pdf_links(li, page_url)
                txt = _clean_pdf_noise(clean(li.get_text()))
                if label == _STAFF_LABEL:
                    txt = _STAFF_CARD_NOISE.sub("", txt).strip()
                if txt:
                    add(f"{i}. {txt}")
                if links_in_li:
                    handle_pdf_links(links_in_li)
            add("")
            return
        # Одиночная ссылка (вне параграфа)
        if el.name == "a":
            href = el.get("href", "").strip()
            txt = clean(el.get_text()) or "документ"
            full = urljoin(page_url, href)
            is_pdf = href.lower().endswith(".pdf")
            is_view = any(
                w in txt.lower()
                for w in ("посмотреть", "открыть", "просмотр", "скачать")
            )
            if (is_pdf or is_view) and full not in handled_pdf_urls:
                handled_pdf_urls.add(full)
                download_pdf(full, session, downloaded)
            return
        if isinstance(el, Tag) and el.name in (
            "div",
            "section",
            "article",
            "main",
            "span",
            "td",
            "th",
        ):
            for child in el.children:
                if isinstance(child, Tag):
                    process_node(child, depth + 1)
            return

    for child in content_el.children:
        process_node(child)
    # Удаление множественных пустых строк
    result = []
    for line in md_lines:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)
    return "\n".join(result).strip()


def page_to_markdown(
    soup, url: str, label: str, session: requests.Session, downloaded: dict
) -> dict:
    """Парсит страницу, извлекает только div.content."""
    content_el = soup.select_one("div.content")
    if not content_el:
        content_el = soup.select_one("[class*='content']")
    if not content_el:
        return {"url": url, "label": label, "markdown": "", "raw_text": ""}

    for tag in content_el.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    raw_text = clean(content_el.get_text(separator=" "))
    body_md = content_to_markdown(content_el, url, session, downloaded, label=label)

    return {
        "url": url,
        "label": label,
        "markdown": body_md,
        "raw_text": raw_text,
    }


def get_inner_links(soup, base_url: str, prefix: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = normalize_url(urljoin(base_url, href))
        if is_internal(full) and not skip_url(full) and full.startswith(prefix):
            links.append(full)
    return links


# Краулер


def crawl(session: requests.Session) -> list[dict]:
    pages: list = []
    visited: set = set()
    used_slugs: set = set()
    downloaded: dict = {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    for url, label, follow in TARGET_PAGES:
        url = normalize_url(url)

        if follow:
            queue: deque = deque([url])
            prefix = url
            section_visited: set = set()
            section_count = 0

            print(f"\n  ── [{label}] (обход раздела)")

            while queue and len(pages) < 500 and section_count < 15:
                cur = queue.popleft()
                if cur in visited or cur in section_visited:
                    continue
                visited.add(cur)
                section_visited.add(cur)
                if skip_url(cur):
                    continue

                print(f"  [{len(pages) + 1}] {cur}")
                soup = get_page(cur, session)
                if not soup:
                    time.sleep(DELAY)
                    continue

                data = page_to_markdown(soup, cur, label, session, downloaded)
                if len(data["raw_text"]) < 50:
                    print("    – пустая страница")
                    time.sleep(DELAY)
                    continue

                pages.append(data)
                save_page(data, used_slugs)
                section_count += 1

                for lnk in get_inner_links(soup, cur, prefix):
                    if lnk not in visited and lnk not in section_visited:
                        queue.append(lnk)

                time.sleep(DELAY)

            if section_count >= 15:
                print(
                    f"    [лимит 15 стр. на раздел достигнут, пропущено ~{len(queue)} URL]"
                )

        else:
            if url in visited:
                continue
            visited.add(url)

            print(f"\n  [{len(pages) + 1}] {url}")
            print(f"  ── [{label}]")
            soup = get_page(url, session)
            if not soup:
                time.sleep(DELAY)
                continue

            data = page_to_markdown(soup, url, label, session, downloaded)
            if len(data["raw_text"]) < 50:
                print("    – пустая страница")
                time.sleep(DELAY)
                continue

            pages.append(data)
            save_page(data, used_slugs)
            time.sleep(DELAY)

    saved_docs = sum(1 for v in downloaded.values() if v is not None)
    skipped = sum(1 for v in downloaded.values() if v is None)
    print(f"\n  PDF скачано (документы) : {saved_docs}")
    print(f"  PDF пропущено (сканы)   : {skipped}")
    return pages


def save_page(data: dict, used_slugs: set):
    # Для страниц «документация» и «питание» .md не создается — только PDF скачаны
    if data.get("label") in _PDF_ONLY_LABELS:
        return
    raw_slug = slugify(data["label"]) or slugify(data["url"].split("/")[-1]) or "page"
    filename = unique_filename(raw_slug, used_slugs) + ".md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data["markdown"])


def main():
    print("=" * 65)
    print("Парсер школьного сайта")
    print(f"Сайт: {BASE_URL}")
    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"\nЦелевых страниц : {len(TARGET_PAGES)}")
    print(f"Папка .md       : ./{OUTPUT_DIR}/")
    print(f"Папка PDF       : ./{DOCS_DIR}/\n")

    # Очищаем папки перед запуском
    for d, ext in [(OUTPUT_DIR, ".md"), (DOCS_DIR, ".pdf")]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(ext):
                    os.remove(os.path.join(d, f))
        os.makedirs(d, exist_ok=True)

    try:
        import pdfminer  # noqa: F401
    except ImportError:
        print("pdfminer.six не установлен. Установите: pip install pdfminer.six")
        print("Без него все PDF будут считаться сканами и пропускаться.\n")

    session = requests.Session()
    session.max_redirects = 10

    pages = crawl(session)

    md_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")])
    pdf_count = len([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")])

    print("\n" + "=" * 65)
    print(f"Готово!")


if __name__ == "__main__":
    main()
