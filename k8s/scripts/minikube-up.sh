#!/usr/bin/env bash
# Поднимает локальный minikube-кластер и разворачивает в нём весь стек School RAG.
# Использование:
#     bash k8s/scripts/minikube-up.sh
#
# Требуется: minikube >= 1.32, kubectl, docker.
set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-school-rag}"
CPUS="${MINIKUBE_CPUS:-4}"
MEMORY="${MINIKUBE_MEMORY:-8192}"
DISK="${MINIKUBE_DISK:-30g}"
DRIVER="${MINIKUBE_DRIVER:-docker}"
IMAGE_TAG="${IMAGE_TAG:-mlops-fastapi:latest}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[1/5] Запускаю minikube (profile=$PROFILE, driver=$DRIVER)..."
if ! minikube status -p "$PROFILE" >/dev/null 2>&1; then
    minikube start \
        -p "$PROFILE" \
        --driver="$DRIVER" \
        --cpus="$CPUS" \
        --memory="$MEMORY" \
        --disk-size="$DISK"
else
    echo "    Кластер $PROFILE уже запущен."
fi

echo "[2/5] Включаю аддоны metrics-server и storage-provisioner..."
minikube addons enable metrics-server -p "$PROFILE" >/dev/null
minikube addons enable storage-provisioner -p "$PROFILE" >/dev/null

echo "[3/5] Собираю Docker-образ FastAPI: $IMAGE_TAG"
(
    cd "$PROJECT_ROOT"
    docker build -t "$IMAGE_TAG" -f Dockerfile .
)

echo "[4/5] Загружаю образ в minikube..."
minikube image load "$IMAGE_TAG" -p "$PROFILE"

echo "[5/5] Применяю Kubernetes-манифесты..."
kubectl apply -k "$PROJECT_ROOT/k8s"

echo
echo "Готово. Жду готовности подов..."
kubectl -n school-rag rollout status deploy/mlflow --timeout=180s || true
kubectl -n school-rag rollout status deploy/ollama --timeout=180s || true
kubectl -n school-rag rollout status deploy/school-rag-api --timeout=300s || true

echo
echo "FastAPI:  $(minikube service school-rag-api -n school-rag --url -p "$PROFILE" | head -n1)"
echo "MLflow :  $(minikube service mlflow -n school-rag --url -p "$PROFILE" | head -n1)"
echo
echo "Подскачайте модели Ollama:"
echo "  kubectl -n school-rag exec deploy/ollama -- ollama pull qwen2.5:7b"
echo "  kubectl -n school-rag exec deploy/ollama -- ollama pull llama3.2"
