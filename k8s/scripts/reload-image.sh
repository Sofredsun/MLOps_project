#!/usr/bin/env bash
# Пересобрать образ FastAPI и перезапустить deployment в minikube.
set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-school-rag}"
IMAGE_TAG="${IMAGE_TAG:-mlops-fastapi:latest}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Сборка образа $IMAGE_TAG..."
docker build -t "$IMAGE_TAG" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"

echo "Загрузка образа в minikube..."
minikube image load "$IMAGE_TAG" -p "$PROFILE"

echo "Перезапуск deployment..."
kubectl -n school-rag rollout restart deploy/school-rag-api
kubectl -n school-rag rollout status deploy/school-rag-api --timeout=300s
