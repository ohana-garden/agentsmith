# AgentSmith Dockerfile
# A working Agent Zero with FalkorDB/Graphiti integration

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget \
    build-essential \
    libffi-dev libssl-dev \
    poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /a0

# Install Python dependencies in stages to avoid conflicts
# Stage 1: Core scientific stack with pinned compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2

# Stage 2: PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Stage 3: ML/NLP packages
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    sentence-transformers==2.2.2 \
    tokenizers==0.15.0 \
    tiktoken==0.5.2

# Stage 4: LLM providers
RUN pip install --no-cache-dir \
    openai==1.12.0 \
    litellm==1.30.0 \
    anthropic==0.18.0

# Stage 5: Web framework and utilities
RUN pip install --no-cache-dir \
    flask[async]==3.0.3 \
    flask-basicauth==0.2.0 \
    python-dotenv==1.0.1 \
    pydantic==2.6.0 \
    nest-asyncio==1.6.0 \
    aiohttp==3.9.3 \
    requests==2.31.0

# Stage 6: Document processing
RUN pip install --no-cache-dir \
    pypdf==4.0.1 \
    pymupdf==1.23.8 \
    beautifulsoup4==4.12.3 \
    html2text==2024.2.26 \
    markdown==3.5.2 \
    markdownify==0.11.6

# Stage 7: Database and graph
RUN pip install --no-cache-dir \
    redis==5.0.1 \
    falkordb==1.0.4 \
    faiss-cpu==1.7.4

# Stage 8: Agent Zero specific
RUN pip install --no-cache-dir \
    langchain==0.1.6 \
    langchain-core==0.1.27 \
    langchain-community==0.0.24 \
    langchain-text-splitters==0.0.1 \
    langchain-unstructured==0.1.0 \
    docker==7.0.0 \
    paramiko==3.4.0 \
    duckduckgo-search==4.4.3 \
    psutil==5.9.8 \
    toml==0.10.2 \
    tomli==2.0.1

# Stage 9: Additional utilities
RUN pip install --no-cache-dir \
    webcolors==1.13 \
    crontab==1.0.1 \
    pathspec==0.12.1 \
    GitPython==3.1.41

# Stage 10: MCP and A2A support
RUN pip install --no-cache-dir \
    fastmcp \
    starlette \
    a2wsgi \
    anyio \
    mcp \
    fasta2a

# Stage 11: Remaining dependencies
RUN pip install --no-cache-dir \
    httpx \
    graphiti-core \
    simpleeval \
    regex \
    PyYAML \
    Pillow \
    cryptography \
    attrs

# Copy application code
COPY . /a0/

# Create necessary directories
RUN mkdir -p /a0/memory /a0/knowledge /a0/logs /a0/tmp

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WEB_UI_PORT=80

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-80}/ || exit 1

# Run - startup check then Agent Zero
CMD python startup_check.py && python run_ui.py --port ${PORT:-80} --host 0.0.0.0
