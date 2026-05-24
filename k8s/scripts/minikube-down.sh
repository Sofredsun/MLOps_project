#!/usr/bin/env bash
# Сносит namespace и (опционально) останавливает minikube.
set -euo pipefail

PROFILE="${MINIKUBE_PROFILE:-school-rag}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Удаляю ресурсы из namespace school-rag..."
kubectl delete -k "$PROJECT_ROOT/k8s" --ignore-not-found=true

if [[ "${STOP_MINIKUBE:-0}" == "1" ]]; then
    echo "Останавливаю minikube profile=$PROFILE..."
    minikube stop -p "$PROFILE"
fi