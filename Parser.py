"""
Парсер сайта http://saki-school2.ucoz.ru/
Фиксированный список страниц + рекурсивный обход /publ/
Контент берется только из div.content
PDF-ссылки («посмотреть») сохраняются как Markdown-ссылки
"""

import requests
from bs4 import BeautifulSoup, Tag
import re
import os
import time
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
from collections import deque

# настройки

BASE_URL = "http://saki-school2.ucoz.ru"
OUTPUT_DIR = "school_knowledge_base/pages"
DELAY = 1.2        # общий лимит по всем страницам

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

# фиксированный список страниц для парсинга
TARGET_PAGES = [
    # (url, label, follow_links)
    ("http://saki-school2.ucoz.ru/index/osnovnye_svedenija/0-58", "основные_сведения", False),
    ("http://saki-school2.ucoz.ru/index/struktura_i_organy_upravlenija_obrazovatelnoj_organizaciej/0-59", "структура_и_органы_управления", False),
    ("http://saki-school2.ucoz.ru/index/rukovodstvo_pedagogicheskij_sostav/0-14", "руководство_педсостав", False),
    ("http://saki-school2.ucoz.ru/index/pedagogicheskij_sostav/0-203", "педагогический_состав", False),
    ("http://saki-school2.ucoz.ru/index/materialno_tekhnicheskoe_obespechenie_i_osnashhennost_obrazovatelnogo_processa/0-56", "материально_техническое",  False),
    ("http://saki-school2.ucoz.ru/index/organizacija_pitanija_v_obrazovatelnoj_organizacii/0-173", "питание", True),   # PDF
    ("http://saki-school2.ucoz.ru/index/dokumentacija/0-42", "документация", True),   # PDF
    ("http://saki-school2.ucoz.ru/index/raspisanie_urokov/0-50",  "расписание_уроков", True),   # PDF
    ("http://saki-school2.ucoz.ru/index/rezhim_raboty_shkoly/0-51", "режим_работы", False),
    ("http://saki-school2.ucoz.ru/index/vserossijskie_proverochnye_raboty/0-76", "всероссийские_проверочные", False),
    ("http://saki-school2.ucoz.ru/index/vneurochnaja_dejatelnost/0-55", "внеурочная_деятельность", True),   # PDF
    ("http://saki-school2.ucoz.ru/publ/", "публикации", True),   # рекурсивный обход
]

SKIP_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico',
    '.zip', '.rar', '.7z', '.mp3', '.mp4', '.avi',
    '.css', '.js', '.woff', '.woff2', '.ttf', '.xml',
)

# утилиты

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment="")).rstrip("/") or "/"

def is_internal(url: str) -> bool:
    parsed = urlparse(url)
    return not parsed.netloc or parsed.netloc == urlparse(BASE_URL).netloc

def skip_url(url: str) -> bool:
    u = url.lower()
    return any(u.endswith(ext) for ext in SKIP_EXTENSIONS)

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or "").strip()

def slugify(text: str) -> str:
    TRANSLIT = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh',
        'з':'z','и':'i','й':'j','к':'k','л':'l','м':'m','н':'n','о':'o',
        'п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'h','ц':'ts',
        'ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
    }
    slug = ''.join(TRANSLIT.get(c, c) for c in text.lower())
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')[:60] or "page"

def unique_filename(base: str, used: set) -> str:
    name, i = base, 2
    while name in used:
        name = f"{base}-{i}"
        i += 1
    used.add(name)
    return name

# http

def get_page(url: str, session: requests.Session):
    try:
        resp = session.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code != 200:
            print(f"[HTTP {resp.status_code}]")
            return None
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None
        return BeautifulSoup(resp.text, "lxml")
    except requests.exceptions.TooManyRedirects:
        print("[redirect loop]")
        return None
    except requests.RequestException as e:
        print(f"[{type(e).__name__}]")
        return None

# парсинг

def parse_table_md(table_tag) -> str:
    headers, rows = [], []
    thead = table_tag.find("thead")
    if thead:
        headers = [clean(c.get_text()) for c in thead.find_all(["th", "td"])]

    header_kw = {"фио","имя","предмет","класс","пн","вт","ср","чт","пт",
                 "урок","время","учитель","должность","№","кабинет",
                 "понедельник","вторник","среда","четверг","пятница"}
    first = True
    tbody = table_tag.find("tbody") or table_tag
    for tr in tbody.find_all("tr"):
        cells = [clean(td.get_text()) for td in tr.find_all(["td","th"])]
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
        lines.append("| " + " | ".join((headers + [""]*col)[:col]) + " |")
        lines.append("|" + " --- |" * col)
    else:
        lines.append("|" + " --- |" * col)
    for row in rows:
        lines.append("| " + " | ".join((row + [""]*col)[:col]) + " |")
    return "\n".join(lines)



def extract_pdf_links(content_el, base_url: str) -> list[dict]:
    """
    находит ссылки на PDF внутри контента
    разделяет «Скачать» и «Посмотреть» и возвращает только «Посмотреть» в формате PDF-ссылки
    дублирующие URL пропускаются
    """
    seen_urls = set()
    pdfs = []
    for a in content_el.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        link_text = clean(a.get_text()) or "документ"
        link_lower = link_text.lower()

        is_pdf_url = href.lower().endswith(".pdf")
        is_view = any(w in link_lower for w in ("посмотреть", "открыть", "просмотр", "view"))
        is_download = any(w in link_lower for w in ("скачать", "download", "загрузить"))

        if is_download and not is_view:
            continue
        if (is_pdf_url or is_view) and full not in seen_urls:
            seen_urls.add(full)
            pdfs.append({"text": link_text, "url": full})
    return pdfs


def content_to_markdown(content_el, page_url: str) -> str:
    """
    конвертирует div.content в Markdown
    """
    md_lines = []
    seen_texts = set()

    def add(line: str):
        #пустые строки пропускаем только если предыдущая тоже пустая
        if not line:
            if md_lines and md_lines[-1] != "":
                md_lines.append("")
            return
        if line not in seen_texts:
            md_lines.append(line)
            seen_texts.add(line)

    #обход только прямых потомков верхнего уровня, чтобы не дублировать вложенный текст
    def process_node(el, depth=0):
        if not isinstance(el, Tag):
            return

        #таблица
        if el.name == "table":
            tbl = parse_table_md(el)
            if tbl:
                add("")
                for line in tbl.splitlines():
                    md_lines.append(line)
                add("")
            return  # не идём внутрь таблицы

        #заголовки
        if el.name in ("h1","h2","h3","h4","h5","h6"):
            level = int(el.name[1])
            txt = clean(el.get_text())
            if txt:
                add("")
                add("#" * level + " " + txt)
                add("")
            return

        #параграф
        if el.name == "p":
            #проверка на ссылки PDF
            links_in_p = extract_pdf_links(el, page_url)
            if links_in_p:
                txt = clean(el.get_text())
                if txt:
                    add(txt)
                for lnk in links_in_p:
                    add(f"  📄 [{lnk['text']}]({lnk['url']})")
                add("")
            else:
                txt = clean(el.get_text())
                if txt:
                    add(txt)
                    add("")
            return

        # Ненумерованный список
        if el.name == "ul":
            for li in el.find_all("li", recursive=False):
                links_in_li = extract_pdf_links(li, page_url)
                txt = clean(li.get_text())
                if links_in_li:
                    add(f"- {txt}")
                    for lnk in links_in_li:
                        add(f"  📄 [{lnk['text']}]({lnk['url']})")
                elif txt:
                    add(f"- {txt}")
            add("")
            return

        # Нумерованный список
        if el.name == "ol":
            for i, li in enumerate(el.find_all("li", recursive=False), 1):
                links_in_li = extract_pdf_links(li, page_url)
                txt = clean(li.get_text())
                if links_in_li:
                    add(f"{i}. {txt}")
                    for lnk in links_in_li:
                        add(f"   📄 [{lnk['text']}]({lnk['url']})")
                elif txt:
                    add(f"{i}. {txt}")
            add("")
            return

        #одиночная ссылка (вне параграфа)
        if el.name == "a":
            href = el.get("href", "").strip()
            txt = clean(el.get_text()) or "документ"
            full = urljoin(page_url, href)
            is_pdf = href.lower().endswith(".pdf")
            is_view = any(w in txt.lower() for w in ("посмотреть","открыть","просмотр","скачать"))
            if is_pdf or is_view:
                add(f"📄 [{txt}]({full})")
            return

        if el.name in ("div", "section", "article", "main", "span", "td", "th"):
            for child in el.children:
                process_node(child, depth + 1)
            return

    for child in content_el.children:
        process_node(child)

    #убираем множественные пустые строки
    result = []
    for line in md_lines:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)

    return "\n".join(result).strip()


def page_to_markdown(soup, url: str, label: str = "") -> dict:
    """Парсит страницу, извлекает только div.content."""

    # только div.content
    content_el = soup.select_one("div.content")
    if not content_el:
        # Запасной вариант — ищем div с классом, содержащим "content"
        content_el = soup.select_one("[class*='content']")
    if not content_el:
        return {"url": url, "markdown": "", "raw_text": ""}

    for tag in content_el.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    raw_text = clean(content_el.get_text(separator=" "))
    body_md = content_to_markdown(content_el, url)

    lines = [body_md]

    return {
        "url": url,
        "label": label,
        "markdown": "\n".join(lines),
        "raw_text": raw_text,
    }


def get_inner_links(soup, base_url: str, prefix: str) -> list[str]:
    """Возвращает внутренние ссылки, начинающиеся с prefix."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = normalize_url(urljoin(base_url, href))
        if (is_internal(full)
                and not skip_url(full)
                and full.startswith(prefix)):
            links.append(full)
    return links

# краулер

def crawl(session: requests.Session) -> list[dict]:
    pages = []
    visited = set()
    used_slugs = set()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for url, label, follow in TARGET_PAGES:
        url = normalize_url(url)

        if follow:
            # Рекурсивный обход внутри раздела
            queue = deque([url])
            prefix = url  # только URL, начинающиеся с этого пути
            section_visited = set()
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

                print(f"  [{len(pages)+1}] {cur}")
                soup = get_page(cur, session)
                if not soup:
                    time.sleep(DELAY)
                    continue

                data = page_to_markdown(soup, cur, label)
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
                print(f"    [лимит 15 стр. на раздел достигнут, пропущено ~{len(queue)} URL]")

        else:
            #одиночная страница
            if url in visited:
                continue
            visited.add(url)

            print(f"\n  [{len(pages)+1}] {url}")
            print(f"  ── [{label}]")
            soup = get_page(url, session)
            if not soup:
                time.sleep(DELAY)
                continue

            data = page_to_markdown(soup, url, label)
            if len(data["raw_text"]) < 50:
                print("    – пустая страница")
                time.sleep(DELAY)
                continue

            pages.append(data)
            save_page(data, used_slugs)
            time.sleep(DELAY)

    return pages


def save_page(data: dict, used_slugs: set):
    """Сохраняет одну страницу как .md файл."""
    raw_slug = slugify(data["label"]) or slugify(data["url"].split("/")[-1]) or "page"
    filename = unique_filename(raw_slug, used_slugs) + ".md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data["markdown"])


# main

def main():
    print("=" * 65)
    print("Парсер школьного сайта → Markdown база знаний (v6)")
    print(f"Сайт: {BASE_URL}")
    print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"\nЦелевых страниц: {len(TARGET_PAGES)}")
    print(f"Папка вывода: ./{OUTPUT_DIR}/\n")

    # Очищаем папку перед запуском
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".md"):
                os.remove(os.path.join(OUTPUT_DIR, f))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = requests.Session()
    session.max_redirects = 10

    pages = crawl(session)

    print("\n" + "=" * 65)
    print(f"  Готово!")
    print(f"  Страниц обработано : {len(pages)}")
    print(f"  .md файлов создано : {len(os.listdir(OUTPUT_DIR))}")
    print(f"  Папка: ./{OUTPUT_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
