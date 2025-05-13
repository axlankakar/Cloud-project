# Use a slim Python base image
FROM python:3.11-slim

# Prevent Python from writing pyc files and using buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential OS packages only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only minimal requirements
COPY requirements.txt .

# Install minimal Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source files
COPY . .

# Expose the default streamlit port
EXPOSE 8501

# Start the Streamlit app (replace `app.py` with your actual filename)
CMD ["streamlit", "run", "app.py"]
