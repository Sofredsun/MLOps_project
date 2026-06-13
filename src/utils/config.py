import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PathConfig:
    """Конфигурация путей"""

    # Исходные данные
    DATA_DIR: Path = PROJECT_ROOT / "data" / "school_knowledge_base"

    # Stage 1 выход
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

    # Stage 2 выход
    CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "chunks"
    TRAIN_DIR: Path = PROJECT_ROOT / "data" / "chunks" / "training_data"
    VAL_DIR: Path = PROJECT_ROOT / "data" / "chunks" / "validation_data"

    # Stage 3 выход
    MODELS_DIR: Path = PROJECT_ROOT / "data" / "models"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_langchain_db"
    EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "models" / "eval_dataset.csv"

    def __post_init__(self):
        """Создает директории если их нет"""
        for path in [
            self.PROCESSED_DIR,
            self.CHUNKS_DIR,
            self.TRAIN_DIR,
            self.VAL_DIR,
            self.MODELS_DIR,
            self.CHROMA_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Stage1Config:
    """Stage 1: Data Loading"""

    # Пути
    paths: PathConfig = field(default_factory=PathConfig)

    # Параметры обработки текста
    REMOVE_PICTURE_BLOCKS: bool = True
    REMOVE_TECHNICAL_INFO: bool = True
    REMOVE_PAGE_NUMBERS: bool = True
    MAX_EMPTY_LINES: int = 2

    # Расширения файлов для обработки
    MARKDOWN_EXTENSIONS: List[str] = field(default_factory=lambda: ["md"])
    PDF_EXTENSIONS: List[str] = field(default_factory=lambda: ["pdf"])

    # Логирование
    VERBOSE: bool = True
    SAVE_STATS: bool = True


@dataclass
class Stage2Config:
    """Stage 2: Data Splitting & Chunking"""

    # Пути
    paths: PathConfig = field(default_factory=PathConfig)

    # Параметры сплиттинга
    PDF_CHUNK_SIZE: int = 800
    PDF_CHUNK_OVERLAP: int = 200
    MARKDOWN_HEADERS: List[tuple] = field(
        default_factory=lambda: [("#", "Header 1"), ("##", "Header 2")]
    )

    # Train/Validation split
    TRAIN_SIZE: float = 0.8  # 80% train, 20% validation
    RANDOM_STATE: int = 42
    SHUFFLE: bool = True

    # Параметры разделения текста
    SEPARATORS: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    # Логирование
    VERBOSE: bool = True
    SAVE_STATS: bool = True


@dataclass
class EmbeddingConfig:
    """Конфигурация для эмбеддингов"""

    # Модель эмбеддингов
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    DEVICE: str = "cpu"  # или "cuda" для GPU
    BATCH_SIZE: int = 32


@dataclass
class LLMConfig:
    """Конфигурация для LLM моделей"""

    # Доступные модели
    AVAILABLE_MODELS: List[str] = field(
        default_factory=lambda: ["qwen2.5:7b", "llama3.2"]
    )

    # Температура для генерации
    TEMPERATURE: float = 0.1

    # Параметры retriever
    RETRIEVER_K: int = 8
    RETRIEVER_SEARCH_TYPE: str = "similarity"

    # Timeout для запросов
    REQUEST_TIMEOUT: int = 60


@dataclass
class MLflowConfig:
    """Конфигурация MLflow"""

    TRACKING_URI: str = "http://localhost:5000"
    EXPERIMENT_NAME: str = "School_RAG_System"
    BACKEND_STORE_URI: str = "sqlite:///mlflow.db"
    ARTIFACTS_DIR: Optional[str] = None

    # Параметры для логирования
    LOG_PARAMS: bool = True
    LOG_METRICS: bool = True
    LOG_ARTIFACTS: bool = True


@dataclass
class Stage3Config:
    """Stage 3: Evaluation"""

    # Пути
    paths: PathConfig = field(default_factory=PathConfig)

    # Модели и конфигурация
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # ChromaDB параметры
    CHROMA_COLLECTION_NAME: str = "school_knowledge_base"
    CHROMA_PERSIST: bool = True

    # Batch параметры
    BATCH_SIZE: int = 10

    # Логирование
    VERBOSE: bool = True
    SAVE_RESULTS: bool = True


@dataclass
class GlobalConfig:
    """Глобальная конфигурация для всего pipeline"""

    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    # Общие параметры
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Инициализация после создания"""
        # Убедиться что все пути согласованы
        paths = PathConfig()
        self.stage1.paths = paths
        self.stage2.paths = paths
        self.stage3.paths = paths


class ConfigFactory:
    """Для создания конфигов с разными settings"""

    @staticmethod
    def create_default() -> GlobalConfig:
        """Создает конфиг с дефолтными параметрами"""
        return GlobalConfig()

    @staticmethod
    def create_development() -> GlobalConfig:
        """Создает конфиг для разработки (быстрые параметры)"""
        config = GlobalConfig()

        # Меньше данных для быстрого тестирования
        config.stage2.TRAIN_SIZE = 0.5  # 50/50 split
        config.stage3.llm.RETRIEVER_K = 4  # Меньше документов
        config.stage3.BATCH_SIZE = 5

        return config

    @staticmethod
    def create_production() -> GlobalConfig:
        """Создает конфиг для production"""
        config = GlobalConfig()

        # Оптимизированные параметры
        config.stage2.PDF_CHUNK_SIZE = 1500  # Больше контекста
        config.stage2.TRAIN_SIZE = 0.9  # 90/10 split
        config.stage3.llm.RETRIEVER_K = 12  # Больше документов
        config.stage3.embedding.DEVICE = "cuda"  # GPU если доступна
        config.LOG_LEVEL = "WARNING"  # Меньше логов

        return config

    @staticmethod
    def create_from_env() -> GlobalConfig:
        """Создает конфиг из переменных окружения"""
        config = GlobalConfig()

        # Читаем переменные окружения
        if os.getenv("ENVIRONMENT") == "production":
            return ConfigFactory.create_production()
        elif os.getenv("ENVIRONMENT") == "development":
            return ConfigFactory.create_development()

        return config


# Дефолтная глобальная конфигурация
CONFIG = GlobalConfig()

# Для быстрого доступа
PATHS = CONFIG.stage1.paths
STAGE1 = CONFIG.stage1
STAGE2 = CONFIG.stage2
STAGE3 = CONFIG.stage3

# Утилиты для создания конфигов в скриптах
__all__ = [
    "CONFIG",
    "PATHS",
    "STAGE1",
    "STAGE2",
    "STAGE3",
    "PathConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "GlobalConfig",
    "ConfigFactory",
]
