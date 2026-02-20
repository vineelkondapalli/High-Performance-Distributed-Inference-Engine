#!/usr/bin/env bash
# Set up Minikube and deploy the inference engine end-to-end.
# Run from the repository root.
#
# Prerequisites: minikube, kubectl, docker
#
# Usage:
#   bash scripts/minikube_setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Starting Minikube ==="
minikube start --memory=4096 --cpus=4 --disk-size=20g

echo ""
echo "=== Pointing Docker CLI at Minikube's daemon ==="
eval "$(minikube docker-env)"

echo ""
echo "=== Copying model into Minikube node ==="
# The Deployment mounts /models from the host node via hostPath.
# We need to create that directory and copy the GGUF file into the node.
minikube ssh "sudo mkdir -p /models"
minikube cp "$REPO_ROOT/models/tinyllama-1.1b-q4.gguf" /models/tinyllama-1.1b-q4.gguf
echo "Model copied."

echo ""
echo "=== Building Docker images inside Minikube ==="
docker build -t cpp-sidecar:latest    "$REPO_ROOT/cpp-sidecar"
docker build -t python-worker:latest  "$REPO_ROOT/python-worker"

echo ""
echo "=== Updating k8s/deployment.yaml to use local image tags ==="
# Patch the deployment to use local images (imagePullPolicy: Never)
kubectl apply -f "$REPO_ROOT/k8s/configmap.yaml"

# Temporarily patch deployment to use local image + Never pull policy
kubectl apply -f - <<EOF
$(sed 's|ghcr.io/GITHUB_USERNAME/cpp-sidecar:latest|cpp-sidecar:latest|g;
       s|ghcr.io/GITHUB_USERNAME/python-worker:latest|python-worker:latest|g;
       s|imagePullPolicy: Always|imagePullPolicy: Never|g' \
       "$REPO_ROOT/k8s/deployment.yaml")
EOF

kubectl apply -f "$REPO_ROOT/k8s/service.yaml"

echo ""
echo "=== Deploying monitoring stack ==="
kubectl apply -f "$REPO_ROOT/k8s/monitoring/prometheus.yaml"
kubectl apply -f "$REPO_ROOT/k8s/monitoring/grafana.yaml"

echo ""
echo "=== Waiting for pods to be ready ==="
kubectl wait deployment/inference-engine --for=condition=available --timeout=300s
kubectl wait deployment/prometheus       --for=condition=available --timeout=60s
kubectl wait deployment/grafana          --for=condition=available --timeout=60s

echo ""
echo "=== Deployment complete! ==="
echo ""
echo "To access the inference engine:"
echo "  minikube service inference-engine"
echo "  # Or: kubectl port-forward svc/inference-engine 8080:8080"
echo ""
echo "To access Grafana (admin/admin):"
echo "  kubectl port-forward svc/grafana 3000:3000"
echo "  # Open http://localhost:3000"
echo ""
echo "To access Prometheus:"
echo "  kubectl port-forward svc/prometheus 9090:9090"
