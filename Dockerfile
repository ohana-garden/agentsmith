# AgentSmith Dockerfile
# A working Agent Zero with FalkorDB/Graphiti integration

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget \
    build-essential \
    libffi-dev libssl-dev \
    poppler-utils tesseract-ocr \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /a0

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Stage 1: PyTorch CPU-only FIRST - this dictates numpy version
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Stage 2: ML/NLP - pinned to Agent Zero versions
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    sentence-transformers==3.0.1 \
    tokenizers \
    tiktoken==0.8.0

# Stage 3: Scientific stack AFTER torch - will get compatible versions
RUN pip install --no-cache-dir \
    scipy \
    scikit-learn \
    faiss-cpu

# Stage 4: LLM providers
RUN pip install --no-cache-dir \
    openai \
    litellm \
    anthropic

# Stage 5: Web framework and utilities
RUN pip install --no-cache-dir \
    flask[async] \
    flask-basicauth \
    python-dotenv \
    pydantic \
    nest-asyncio \
    aiohttp \
    requests

# Stage 6: Document processing
RUN pip install --no-cache-dir \
    pypdf \
    pymupdf \
    beautifulsoup4 \
    html2text \
    markdown \
    markdownify

# Stage 7: Database and graph
RUN pip install --no-cache-dir \
    redis \
    falkordb

# Stage 8: Langchain ecosystem (pinned to Agent Zero versions)
RUN pip install --no-cache-dir \
    langchain-core==0.3.49 \
    langchain-community==0.3.19 \
    langchain-text-splitters

# Stage 9: Agent tools
RUN pip install --no-cache-dir \
    docker \
    paramiko \
    duckduckgo-search \
    psutil \
    toml \
    tomli

# Stage 10: Additional utilities
RUN pip install --no-cache-dir \
    webcolors \
    crontab \
    pathspec \
    GitPython

# Stage 11: MCP and A2A support (pinned versions for API compatibility)
RUN pip install --no-cache-dir \
    fastmcp==2.3.4 \
    mcp==1.13.1 \
    fasta2a==0.5.0 \
    a2wsgi==1.10.8 \
    starlette \
    anyio

# Stage 12: Remaining dependencies
RUN pip install --no-cache-dir \
    httpx \
    graphiti-core \
    simpleeval \
    regex \
    PyYAML \
    Pillow \
    cryptography \
    attrs \
    exchangelib \
    imapclient \
    kokoro \
    soundfile \
    browser-use \
    flaredantic \
    inputimeout \
    pdf2image \
    pytesseract \
    pytz \
    werkzeug \
    yt-dlp

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
