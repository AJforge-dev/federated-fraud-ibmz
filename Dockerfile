# ==============================================================================
# Dockerfile: Bank Node for Confidential Federated Learning on IBM Z & LinuxONE
# ==============================================================================
# Target Platform: linux/s390x (IBM z16 / LinuxONE Emperor 4 & Rockhopper 4)
# Enclave Target: IBM Secure Execution for Linux (confidential guest VM)
# Base Image: Official Python 3.11 slim image supporting multi-arch (amd64, arm64, s390x)
#
# NOTE: In production on IBM LinuxONE, this container runs inside an encrypted,
# hardware-isolated IBM Secure Execution guest where memory and CPU state are
# protected by CPACF hardware keys from host hypervisors and administrators.
# ==============================================================================

FROM --platform=$TARGETPLATFORM python:3.11-slim

LABEL maintainer="IBM Z Datathon 2026 Team"
LABEL description="Confidential Federated Learning Bank Node for Fraud Detection"
LABEL challenge="AI Secured: Innovation Without Exposure"

# Prevent interactive prompts during build and disable Python bytecode/buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal OS dependencies required for C-extensions and numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root unprivileged service user for financial data isolation
RUN useradd -m -u 1001 banknode
WORKDIR /app

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and documentation
COPY src/ /app/src/
COPY docs/ /app/docs/
COPY app.py /app/app.py

# Create required runtime directories and set ownership
RUN mkdir -p /app/data /app/results && \
    chown -R banknode:banknode /app

USER banknode

# Expose ports: 8501 for Streamlit Dashboard, 8080 for Secure Enclave RPC
EXPOSE 8501 8080

# Health check to ensure node responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.__version__)" || exit 1

# Default command runs the federated benchmark suite
CMD ["python", "-m", "src.experiments"]
