## School RAG — MLOps-система школьного ИИ-ассистента

**School RAG** - это воспроизводимая MLOps-система для LLM-ассистента на базе технологии RAG. Ассистент отвечает на вопросы по базе знаний школы (расписание, питание, правила, контакты), а вокруг него построен полный production-цикл: поступление новых данных, обнаружение деградации модели, переобучение, обновление сервиса и мониторинг качества через веб-интерфейс.

---

## Основные возможности

- RAG-инференс: поиск релевантных фрагментов в ChromaDB и генерация ответа локальной LLM (Ollama)
- Трекинг экспериментов и Model Registry в MLflow
- Обнаружение data drift (сдвиг тематики запросов по эмбеддингам)
- Обнаружение concept/target drift (падение качества по фидбэку и RAGAS-метрикам)
- Генерация HTML-отчётов о дрейфе
- Ручное и программное переобучение (переиндексация базы знаний)
- Мониторинг метрик в Prometheus и Grafana
- Веб-интерфейс на Streamlit: инференс, история предсказаний, источники, лайки/дизлайки, уведомления о дрейфе, кнопка переобучения
- Версионирование данных и модели через DVC

---

## Архитектура

Проект построен по слоистой архитектуре, код разбит на изолированные этапы с чёткими входами и выходами:

- **API слой** - FastAPI: инференс, фидбэк, мониторинг дрейфа, переобучение, метрики
- **UI слой** - Streamlit: чат с ассистентом, панель алертов и управления
- **Слой данных** - пайплайн из трёх этапов: загрузка и очистка документов, чанкинг и train/val split, построение ChromaDB и оценка качества

Такой подход позволяет запускать каждый шаг независимо в автоматизированных CI/CD-пайплайнах.

---

## Используемый стек

- Python 3.10+
- FastAPI, Uvicorn
- Streamlit
- LangChain
- Ollama (qwen2.5:7b, llama3.2)
- ChromaDB
- HuggingFace эмбеддинги (multilingual-e5)
- MLflow
- DVC
- Prometheus, Grafana
- Docker, Docker Compose
- Kubernetes / Minikube (Kustomize)
- Argo CD
- GitHub Actions
- Isort, Black, Flake8, Pytest

---

## Развертывание

Проект можно запускать:

- локально (uvicorn + streamlit)
- в Docker-контейнерах через Docker Compose (для отладки)
- в Kubernetes / Minikube (production)

Continuous Delivery реализован через Argo CD по GitOps-подходу: Argo CD следит за директорией `k8s` в репозитории и приводит состояние кластера к git.

---

## Цель проекта

- Реализовать полный цикл эксплуатации ML-модели в production-условиях
- Продемонстрировать практики MLOps: версионирование кода (Git Flow, conventional commits) и данных (DVC), трекинг экспериментов (MLflow)
- Показать практики CI (линтинг, тесты, сборка и упаковка в docker-образ с сохранением в GHCR)
- Показать практики CD (деплой в Kubernetes через Argo CD)
- Реализовать мониторинг качества и обнаружение дрейфа с автоматическим/ручным переобучением

---

## DVC

- Данные (`data/`) и собранная база ChromaDB (`chroma_langchain_db/`) версионируются через DVC; указатели — файлы `data.dvc` и `chroma_langchain_db.dvc`
- Remote настроен на Google Drive (service account)
- Чтобы скачать данные и базу знаний: `dvc pull`
- Чтобы выгрузить изменения: `dvc push`

---

## CI/CD Pipeline

### Continuous Integration (GitHub Actions)

При каждом пуше в `feat/CI-CD` (для тестирования) и `main`, а также при PR в `develop` и `main` автоматически запускается пайплайн из четырёх джобов:

1. **Lint** — проверка форматирования (Black, isort, Flake8)
2. **Tests** — запуск тестов с покрытием (pytest, coverage)
3. **Build Docker** — сборка и публикация образа в GHCR (`ghcr.io/bitterrch/mlops_project`)
4. **Deploy to k8s (kind)** — развёртывание всего стека в изолированном Kubernetes-кластере

### Continuous Delivery (Argo CD)

Argo CD следит за веткой `main` и директорией `k8s/`. При мерже в `main`:
- CI собирает образ и публикует его с тегом `latest` в GHCR
- Argo CD обнаруживает новый коммит и автоматически синхронизирует кластер
- Все сервисы обновляются по GitOps-подходу без ручного вмешательства

---

## Запуск проекта

### Локальный запуск

Требуется Python 3.10+ и установленный [Ollama](https://ollama.com).

1. Клонируйте репозиторий и установите зависимости:
```bash
git clone https://github.com/bitterrch/mlops_project.git
cd mlops_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Подтяните данные и базу знаний из DVC:
```bash
dvc pull
```
3. Скачайте модели Ollama:
```bash
ollama pull qwen2.5:7b
ollama pull llama3.2
```
4. Запустите MLflow (опционально), API и UI:
```bash
mlflow ui # http://localhost:5000
uvicorn api:app --reload # http://localhost:8000
streamlit run app.py # http://localhost:8501
```

### Запуск через Docker Compose

Поднимает весь стек одной командой (для отладки):
```bash
docker compose up -d --build
```
После старта подтяните модели в контейнер Ollama:
```bash
docker exec -it ollama ollama pull qwen2.5:7b
docker exec -it ollama ollama pull llama3.2
```

Сервисы:
- FastAPI: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Запуск в Minikube

```bash
bash k8s/scripts/minikube-up.sh
```
Скрипт поднимает кластер, собирает образ FastAPI, загружает его в Minikube и применяет манифесты в namespace `school-rag`. Получить адреса сервисов:
```bash
minikube service school-rag-api -n school-rag --url
minikube service school-rag-ui  -n school-rag --url
```
Подтянуть модели в Ollama внутри кластера:
```bash
kubectl -n school-rag exec deploy/ollama -- ollama pull qwen2.5:7b
kubectl -n school-rag exec deploy/ollama -- ollama pull llama3.2
```

### CD через Argo CD

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Приложение будет доступно по адресу:
http://localhost:8000

Swagger-документация:
http://localhost:8000/docs

Redoc-документация:
http://localhost:8000/redoc
