# Корпоративный RAG-ассистент
Этот репозиторий содержит материалы проекта по созданию воспроизводимой MLOps-системы для LLM-ассистента на базе технологии RAG . Проект реализует полный цикл эксплуатации ML-модели

**Текущий этап:** сбор данных и их анализ

## Инструкция по запуску
1. Склонируйте репозиторий и в терминале перейдите в папку `mlops_project`
    ```bash
    git clone https://github.com/bitterrch/mlops_project.git
    cd mlops_project
2. Создайте и активируйте виртуальное окружение
    ```bash
    python -m venv venv
    venv\Scripts\activate
3. Установите зависимости 
    ```bash
    pip install -r requirements.txt
4. Создайте ядро для Jupyter Notebook
    ```bash
    python -m ipykernel install --user --name mlops_project
5. Чтобы запустить процесс парсинга сайта и наполнить папку `school_knowledge_database` свежими файлами, выполните
    ```bash
    python Parser.py
6. Чтобы посмотреть результаты анализа текстов, запустите Jupyter:
    ```bash
    jupyter notebook
    ```
    Затем в Jupyter-е откройте `jupyter/01_eda.ipynb`, выберите ядро `mlops_project` и выполните весь код