"""
MLOps Pipeline Orchestrator
Запускает все stages последовательно
"""

import sys
import subprocess
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STAGES = [
    (1, "stages/download_data.py", "Data Loading"),
    (2, "stages/splitter.py", "Data Processing"),
    (3, "stages/evaluation.py", "Model Evaluation"),
]


def run_stage(stage_num: int, script: str, description: str) -> bool:
    """Запускает отдельный этап"""
    logger.info("=" * 70)
    logger.info(f"STAGE {stage_num}: {description}")
    logger.info("=" * 70)

    result = subprocess.run([sys.executable, script], cwd="src")

    if result.returncode != 0:
        logger.error(f"STAGE {stage_num} FAILED")
        return False

    logger.info(f"STAGE {stage_num} COMPLETED\n")
    return True


def main():
    """Запускает полный pipeline"""
    for stage_num, script, description in STAGES:
        if not run_stage(stage_num, script, description):
            logger.error(f"Pipeline остановлен на этапе {stage_num}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
