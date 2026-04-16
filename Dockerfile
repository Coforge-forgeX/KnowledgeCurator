FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by Playwright(+ git REQUIRED for pip git dependencies)
RUN apt-get update && apt-get install -y \
    git \
    wget curl unzip \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libasound2 libpangocairo-1.0-0 libpango-1.0-0 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy entire repo
COPY . /app

# Set Python path so src is detectable
ENV PYTHONPATH=/app/src

# Upgrade pip
RUN pip install --upgrade pip

# Install custom packages inside .venv
RUN pip install -e . --force-reinstall

# Install Playwright browsers
RUN playwright install --with-deps chromium

# Install Playwright browsers
RUN playwright install --with-deps chromium

# Expose FastAPI port
EXPOSE 9000

# Run FastAPI app
#CMD ["uvicorn", "kbcurrator.server.main:http_app", "--host", "0.0.0.0", "--port", "9000"]
CMD ["sh", "-c", "uvicorn kbcurator.server.main:http_app --host 0.0.0.0 --port ${PORT:-9000}"]
