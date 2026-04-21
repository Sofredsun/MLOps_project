import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pymupdf4llm
from langchain_core.documents import Document

from utils.config import STAGE1

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

"""
Stage 1: Data Loading and Preprocessing
Скрипт для загрузки документов из папки data/ и их предварительной обработки.
Входные данные: PDF и Markdown файлы
Выходные данные: Обработанные документы
"""

DATA_DIR = STAGE1.paths.DATA_DIR
OUTPUT_DIR = STAGE1.paths.PROCESSED_DIR


def clean_text(text: str) -> str:
    """Функция для очистки текста"""
    # Убираем лишние переносы строк внутри абзацев
    text = re.sub(r"(?<=[^\\.!?])\n+(?=[а-яёa-z])", " ", text)
    # убираем множественные пробелы
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_school_md(text: str) -> str:
    """
    Специализированная очистка для школьных документов в формате Markdown.
    Удаляет служебную информацию, номера страниц, лишние блоки.
    """
    # 1. Удаляем блоки с описанием пропущенных картинок
    text = re.sub(r"==> picture \[.*?\] intentionally omitted <==", "", text)
    text = re.sub(
        r"----- Start of picture text -----.*?----- End of picture text -----",
        "",
        text,
        flags=re.DOTALL,
    )

    # 2. Удаляем техническую информацию о цифровой подписи
    text = re.sub(
        r"РАССМОТРЕНО.*?Дата: \d{4}\.\d{2}\.\d{2}.*?\+03\'00\'",
        "",
        text,
        flags=re.DOTALL,
    )

    # 3. Удаляем номера страниц (например, _ 2 _, _ 3 _)
    text = re.sub(r"_\s\d+\s_", "", text)

    # 4. Убираем множественные пустые строки (оставляем максимум две для структуры)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Убираем лишние пробелы в начале и конце строк
    text = "\n".join([line.strip() for line in text.split("\n")])

    return text.strip()


def load_documents() -> Tuple[List[Document], List[Document]]:
    """
    Загружает все документы из DATA_DIR.

    Returns:
        Tuple[List[Document], List[Document]]: (markdown документы, PDF документы)
    """
    data_md = []
    data_pdf = []

    if not os.path.exists(DATA_DIR):
        print(f"Папка {DATA_DIR} не найдена")
        return data_md, data_pdf

    # Обходим всю структуру папок
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            extension = file.split(".")[-1].lower()
            text = ""

            try:
                if extension == "md":
                    print(f"Загружаю: {file}")
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text:
                            # Очищаем текст
                            text = clean_school_md(text)
                            data_md.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": file,
                                        "path": path,
                                        "format": extension,
                                        "size_bytes": os.path.getsize(path),
                                    },
                                )
                            )

                elif extension == "pdf":
                    print(f"Конвертирую PDF: {file}")
                    # Используем pymupdf4llm для конвертации PDF -> Markdown
                    text = pymupdf4llm.to_markdown(path)
                    text = clean_school_md(text)
                    if text:
                        data_pdf.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file,
                                    "path": path,
                                    "format": extension,
                                    "size_bytes": os.path.getsize(path),
                                },
                            )
                        )

            except Exception as e:
                print(f"Ошибка при чтении файла {file}: {e}")

    return data_md, data_pdf


def save_documents_info(data_md: List[Document], data_pdf: List[Document]) -> None:
    """Сохраняет информацию о загруженных документах в JSON для отладки"""
    info = {
        "total_documents": len(data_md) + len(data_pdf),
        "markdown_count": len(data_md),
        "pdf_count": len(data_pdf),
        "markdown_files": [
            {
                "source": doc.metadata["source"],
                "size_bytes": doc.metadata["size_bytes"],
                "content_length": len(doc.page_content),
            }
            for doc in data_md
        ],
        "pdf_files": [
            {
                "source": doc.metadata["source"],
                "size_bytes": doc.metadata["size_bytes"],
                "content_length": len(doc.page_content),
            }
            for doc in data_pdf
        ],
    }

    info_path = os.path.join(OUTPUT_DIR, "data_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\nИнформация о документах сохранена в {info_path}")


def main():
    """Основная функция для этапа загрузки данных"""
    print("=" * 60)
    print("STAGE 1: Data Loading and Preprocessing")
    print("=" * 60)

    # Загружаем документы
    print(f"\nЗагружаю документы из: {DATA_DIR}\n")
    data_md, data_pdf = load_documents()

    # Выводим статистику
    print("\n" + "=" * 60)
    print(f"Всего загружено документов: {len(data_md) + len(data_pdf)}")
    print(f"   - Markdown файлы: {len(data_md)}")
    print(f"   - PDF файлы: {len(data_pdf)}")
    print("=" * 60)

    # Сохраняем информацию
    save_documents_info(data_md, data_pdf)

    # Сохраняем документы для следующего этапа
    documents_path = os.path.join(OUTPUT_DIR, "documents.json")
    documents_data = {
        "markdown": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in data_md
        ],
        "pdf": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in data_pdf
        ],
    }

    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(documents_data, f, ensure_ascii=False, indent=2)

    print(f"Обработанные документы сохранены в {documents_path}")

    return data_md, data_pdf


if __name__ == "__main__":
    data_md, data_pdf = main()
