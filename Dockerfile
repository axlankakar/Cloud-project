# Base lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Avoid interactive prompts and unnecessary cache
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install required system packages (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements to optimize layers
COPY requirements.txt .

# Install only core required packages (CPU-only)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
