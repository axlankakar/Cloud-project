# Use a smaller Python base image
FROM python:3.12-slim

# Avoid writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies step-by-step to reduce memory spikes
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install core dependencies separately to reduce Docker layer size
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    faiss-cpu \
    torch \
    transformers

# Now install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
