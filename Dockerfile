FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p cache lancedb knowledge/docling && \
    chmod -R 755 cache lancedb knowledge

# Create a non-root user for security (optional but recommended)
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port
EXPOSE $PORT

# Simple health check that actually works with Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Create Streamlit config directory and file
RUN mkdir -p ~/.streamlit
RUN echo '[server]\nport = '$PORT'\naddress = "0.0.0.0"\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n[browser]\ngatherUsageStats = false' > ~/.streamlit/config.toml

# Run the application
CMD ["streamlit", "run", "app.py"]