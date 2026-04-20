"""
Stage 2: Data Splitting and Chunking
Скрипт для разделения документов на чанки и их распределения на train/validation наборы.
Входные данные: Обработанные документы от stage 1
Выходные данные: Training Data и Validation Data
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.config import STAGE2, PATHS

PDF_CHUNK_SIZE = STAGE2.PDF_CHUNK_SIZE
PDF_CHUNK_OVERLAP = STAGE2.PDF_CHUNK_OVERLAP
TRAIN_TEST_SPLIT = STAGE2.TRAIN_SIZE
RANDOM_STATE = STAGE2.RANDOM_STATE
MARKDOWN_HEADERS = STAGE2.MARKDOWN_HEADERS

INPUT_FILE = PATHS.PROCESSED_DIR / "documents.json"
OUTPUT_DIR = PATHS.CHUNKS_DIR
TRAIN_DIR = PATHS.TRAIN_DIR
VAL_DIR = PATHS.VAL_DIR


def load_documents_from_json() -> Tuple[List[Document], List[Document]]:
    """Загружает документы из JSON файла (выход от stage 1)"""
    documents_path = PATHS.PROCESSED_DIR / "documents.json"

    if not os.path.exists(documents_path):
        raise FileNotFoundError(
            f"Файл {documents_path} не найден. Сначала запустите stage 1."
        )

    with open(documents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_md = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in data["markdown"]
    ]

    data_pdf = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in data["pdf"]
    ]

    return data_md, data_pdf


def split_documents_into_chunks(
    data_md: List[Document], data_pdf: List[Document]
) -> List[Document]:
    """
    Разделяет документы на чанки используя специализированные сплиттеры.

    Args:
        data_md: Markdown документы
        data_pdf: PDF документы (уже в формате Markdown)

    Returns:
        List[Document]: Список всех чанков
    """

    # Инициализируем сплиттеры
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)

    all_chunks = []

    # Обработка Markdown документов с сохранением структуры заголовков
    print("\nОбработка Markdown документов...")
    for doc in tqdm(data_md, desc="Markdown splitting"):
        try:
            md_chunks = md_splitter.split_text(doc.page_content)

            # Обновляем метаданные каждого чанка
            for chunk in md_chunks:
                # Сохраняем оригинальные метаданные документа
                chunk.metadata.update(doc.metadata)

            # Дополнительный сплиттинг для больших секций
            final_chunks = pdf_splitter.split_documents(md_chunks)
            all_chunks.extend(final_chunks)

        except Exception as e:
            print(f"️  Ошибка обработки документа {doc.metadata.get('source')}: {e}")

    # Обработка PDF документов (уже в формате Markdown)
    print("\nОбработка PDF документов...")
    pdf_chunks = pdf_splitter.split_documents(data_pdf)
    all_chunks.extend(pdf_chunks)

    return all_chunks


def split_chunks_train_test(
    chunks: List[Document], train_size: float = TRAIN_TEST_SPLIT
) -> Tuple[List[Document], List[Document]]:
    """
    Разделяет чанки на обучающую и валидационную выборки.

    Args:
        chunks: Все чанки
        train_size: Доля для обучения (0.8 = 80% train, 20% val)

    Returns:
        Tuple: (train_chunks, validation_chunks)
    """

    train_chunks, val_chunks = train_test_split(
        chunks, train_size=train_size, random_state=RANDOM_STATE, shuffle=True
    )

    return train_chunks, val_chunks


def save_chunks_to_files(
    chunks: List[Document], output_dir: str, dataset_name: str
) -> Dict[str, Any]:
    """
    Сохраняет чанки в файлы для дальнейшего использования.

    Args:
        chunks: Список чанков для сохранения
        output_dir: Директория для сохранения
        dataset_name: Название датасета (для логов)

    Returns:
        Dict с статистикой
    """

    # Сохраняем в pickle (бинарный формат для Python)
    pickle_path = os.path.join(output_dir, f"{dataset_name}_chunks.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(chunks, f)

    # Сохраняем метаинформацию в JSON
    stats = {
        "total_chunks": len(chunks),
        "sources": list(
            set(chunk.metadata.get("source", "Unknown") for chunk in chunks)
        ),
        "avg_chunk_size": (
            sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            if chunks
            else 0
        ),
        "total_characters": sum(len(chunk.page_content) for chunk in chunks),
    }

    json_path = os.path.join(output_dir, f"{dataset_name}_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"{dataset_name.capitalize()} сохранено:")
    print(f"   - Файл: {pickle_path}")
    print(f"   - Количество чанков: {stats['total_chunks']}")
    print(f"   - Среднее размер чанка: {stats['avg_chunk_size']:.0f} символов")
    print(f"   - Всего символов: {stats['total_characters']}")

    return stats


def main():
    """Основная функция для этапа разделения данных"""
    print("=" * 60)
    print("STAGE 2: Data Splitting and Chunking")
    print("=" * 60)

    # Загружаем обработанные документы
    print("\nЗагружаю обработанные документы...")
    try:
        data_md, data_pdf = load_documents_from_json()
        print(f"Загружено документов: {len(data_md) + len(data_pdf)}")
        print(f"   - Markdown: {len(data_md)}")
        print(f"   - PDF: {len(data_pdf)}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return

    # Разделяем документы на чанки
    print("\n  Разделяю документы на чанки...")
    all_chunks = split_documents_into_chunks(data_md, data_pdf)
    print(f"Создано чанков: {len(all_chunks)}")

    # Разделяем чанки на train/val
    print(
        f"\n Разделяю на training ({int(TRAIN_TEST_SPLIT*100)}%) и validation ({int((1-TRAIN_TEST_SPLIT)*100)}%)..."
    )
    train_chunks, val_chunks = split_chunks_train_test(
        all_chunks, train_size=TRAIN_TEST_SPLIT
    )

    # Сохраняем результаты
    print("\n Сохраняю результаты...")
    train_stats = save_chunks_to_files(train_chunks, TRAIN_DIR, "training")
    print()
    val_stats = save_chunks_to_files(val_chunks, VAL_DIR, "validation")

    # Сохраняем общую статистику
    summary = {
        "total_chunks": len(all_chunks),
        "training": train_stats,
        "validation": val_stats,
        "split_ratio": TRAIN_TEST_SPLIT,
        "chunk_size": PDF_CHUNK_SIZE,
        "chunk_overlap": PDF_CHUNK_OVERLAP,
    }

    summary_path = os.path.join(OUTPUT_DIR, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("ЭТАП 2 ЗАВЕРШЕН")
    print("=" * 60)
    print(f"Результаты сохранены в: {OUTPUT_DIR}")

    return train_chunks, val_chunks


if __name__ == "__main__":
    train_chunks, val_chunks = main()
