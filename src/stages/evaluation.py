"""
Stage 3: Evaluation and Model Testing
Скрипт для создания эмбеддингов, инициализации ChromaDB,
генерации ответов и оценки качества модели.
"""

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.config import STAGE3, PATHS

TRAIN_DIR = PATHS.TRAIN_DIR
VAL_DIR = PATHS.VAL_DIR
OUTPUT_DIR = PATHS.MODELS_DIR
CHROMA_DIR = PATHS.CHROMA_DIR
EMBEDDING_MODEL = STAGE3.embedding.MODEL_NAME
AVAILABLE_LLM_MODELS = STAGE3.llm.AVAILABLE_MODELS
RETRIEVER_K = STAGE3.llm.RETRIEVER_K
EVAL_DATASET_PATH = PATHS.EVAL_DATASET_PATH

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("School_RAG_Evaluation")


def compute_semantic_similarity(
    embeddings_model: HuggingFaceEmbeddings, text1: str, text2: str
) -> float:
    emb1 = embeddings_model.embed_query(text1)
    emb2 = embeddings_model.embed_query(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0])


def compute_faithfulness(answer: str, context: str) -> float:
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def load_chunks(chunk_dir: Path, chunk_type: str = "training") -> List[Document]:
    pickle_path = chunk_dir / f"{chunk_type}_chunks.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Файл {pickle_path} не найден. Сначала запустите stage 2."
        )

    with open(pickle_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Загружено {len(chunks)} чанков из {chunk_type} набора")
    return chunks


def load_eval_dataset(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Файл датасета {csv_path} не найден.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=";")
    required_cols = {"question", "ground_truth_answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"В CSV должны быть колонки: {required_cols}")
    print(f"Загружено {len(df)} вопросов для оценки")
    return df


def initialize_embeddings() -> HuggingFaceEmbeddings:
    print(f"\nИнициализирую модель эмбеддингов: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def create_chroma_db(
    chunks: List[Document], embeddings: HuggingFaceEmbeddings
) -> Chroma:
    print(f"\nСоздание ChromaDB в {CHROMA_DIR}...")
    db = Chroma(
        collection_name="school_knowledge_base",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    batch_size = 20
    for i in tqdm(range(0, len(chunks), batch_size), desc="Загрузка чанков в ChromaDB"):
        batch = chunks[i : i + batch_size]
        try:
            db.add_documents(batch)
        except Exception as e:
            print(f"\n⚠️  Ошибка на батче {i // batch_size}: {e}")
    print(f"ChromaDB создана с {len(chunks)} чанками")
    return db


def test_rag_system(
    db: Chroma,
    eval_df: pd.DataFrame,
    embeddings: HuggingFaceEmbeddings,
    models: List[str] = None,
) -> List[Dict[str, Any]]:
    if models is None:
        models = AVAILABLE_LLM_MODELS

    results = []
    template = """Вы — экспертный аналитик базы знаний школы. 
Ваша цель: найти ответ на вопрос в предоставленных фрагментах документов.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ИНСТРУКЦИЯ:
Проанализируй контекст. Если ответ найден частично, напиши то, что удалось найти.
ОТВЕТ:"""

    prompt = ChatPromptTemplate.from_template(template)
    # Исправлен пробел в ключе "k"
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )

    for model_name in models:
        print(f"\nИнициализирую модель: {model_name}")
        try:
            llm = OllamaLLM(model=model_name, temperature=0.1)
            chain = prompt | llm
        except Exception as e:
            print(f"Ошибка инициализации {model_name}: {e}")
            continue

        with mlflow.start_run(run_name=f"RAG_Test_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("embedding_model", EMBEDDING_MODEL)
            mlflow.log_param("retriever_k", RETRIEVER_K)
            mlflow.log_param("num_questions", len(eval_df))

            sim_list, faith_list, lat_list = [], [], []

            for idx, row in tqdm(
                eval_df.iterrows(), total=len(eval_df), desc=f"Testing {model_name}"
            ):
                question = row["question"]
                ground_truth = row["ground_truth_answer"]

                start_time = time.time()
                try:
                    retrieved_docs = retriever.invoke(question)

                    context_text = "\n\n".join(
                        [
                            f"[Источник: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                            for doc in retrieved_docs
                        ]
                    )

                    response = chain.invoke(
                        {"context": context_text, "question": question}
                    )
                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    answer = answer.strip()

                    latency = time.time() - start_time

                    sim = compute_semantic_similarity(embeddings, answer, ground_truth)
                    faith = compute_faithfulness(answer, context_text)

                    sim_list.append(sim)
                    faith_list.append(faith)
                    lat_list.append(latency)

                    # Логируем с шагом, чтобы не перезаписывать метрики
                    mlflow.log_metric("semantic_similarity", sim, step=idx)
                    mlflow.log_metric("faithfulness", faith, step=idx)
                    mlflow.log_metric("latency_seconds", latency, step=idx)
                    mlflow.log_metric("context_length", len(context_text), step=idx)

                    results.append(
                        {
                            "model": model_name,
                            "question": question,
                            "ground_truth": ground_truth,
                            "answer": answer,
                            "semantic_similarity": round(sim, 3),
                            "faithfulness": round(faith, 3),
                            "latency_seconds": round(latency, 3),
                            "context_length": len(context_text),
                            "timestamp": datetime.now().isoformat(),
                            "error": False,
                        }
                    )
                except Exception as e:
                    print(f"Ошибка при обработке вопроса: {e}")
                    mlflow.log_metric("error_flag", 1, step=idx)
                    results.append(
                        {
                            "model": model_name,
                            "question": question,
                            "ground_truth": ground_truth,
                            "answer": f"ERROR: {str(e)}",
                            "semantic_similarity": 0.0,
                            "faithfulness": 0.0,
                            "latency_seconds": -1.0,
                            "context_length": 0,
                            "timestamp": datetime.now().isoformat(),
                            "error": True,
                        }
                    )
            valid_lat = [l for l in lat_list if l > 0]
            mlflow.log_metric("avg_semantic_similarity", float(np.mean(sim_list)))
            mlflow.log_metric("avg_faithfulness", float(np.mean(faith_list)))
            mlflow.log_metric("min_similarity", float(np.min(sim_list)))
            mlflow.log_metric(
                "avg_latency", float(np.mean(valid_lat)) if valid_lat else 0.0
            )

        print(f"{model_name}: тестирование завершено")

    return results


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {}
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    for model, model_results in by_model.items():
        errors = sum(1 for r in model_results if r.get("error"))
        total = len(model_results)
        valid = [r for r in model_results if not r["error"]]

        metrics[model] = {
            "total_tests": total,
            "successful": total - errors,
            "errors": errors,
            "error_rate": round(errors / total, 3) if total > 0 else 0,
            "avg_semantic_similarity": (
                round(np.mean([r["semantic_similarity"] for r in valid]), 3)
                if valid
                else 0
            ),
            "avg_faithfulness": (
                round(np.mean([r["faithfulness"] for r in valid]), 3) if valid else 0
            ),
            "avg_latency": round(
                np.mean(
                    [r["latency_seconds"] for r in valid if r["latency_seconds"] > 0]
                ),
                3,
            ),
            "min_similarity": (
                round(min([r["semantic_similarity"] for r in valid]), 3) if valid else 0
            ),
        }
    return metrics


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_csv = OUTPUT_DIR / "evaluation_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False, encoding="utf-8")
    print(f"Результаты сохранены в {results_csv}")

    metrics_json = OUTPUT_DIR / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Метрики сохранены в {metrics_json}")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    for model, m in metrics.items():
        print(f"\n🔹 {model}")
        print(
            f"   Успешно: {m['successful']}/{m['total_tests']} (ошибок: {m['error_rate'] * 100:.1f}%)"
        )
        print(
            f"   Avg Similarity: {m['avg_semantic_similarity']:.3f} | Min: {m['min_similarity']:.3f}"
        )
        print(f"   Avg Faithfulness: {m['avg_faithfulness']:.3f}")
        print(f"   Avg Latency: {m['avg_latency']:.3f}s")


def main():
    print("=" * 60)
    print("STAGE 3: Evaluation and Model Testing")
    print("=" * 60)

    print("\nЗагружаю данные...")
    train_chunks = load_chunks(TRAIN_DIR, "training")
    eval_df = load_eval_dataset(EVAL_DATASET_PATH)

    embeddings = initialize_embeddings()
    db = create_chroma_db(train_chunks, embeddings)

    print("\nТестирование RAG системы...")
    results = test_rag_system(db, eval_df, embeddings)

    print("\nВычисление метрик...")
    metrics = calculate_metrics(results)

    print("\nСохранение результатов...")
    save_results(results, metrics)

    print("\n" + "=" * 60)
    print("ЭТАП 3 ЗАВЕРШЕН")
    print("=" * 60)
    return db, results, metrics


if __name__ == "__main__":
    db, results, metrics = main()
