# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a working directory inside the container
WORKDIR /app

# Copy only dependency files first to leverage Docker layer caching
# (We'll copy requirements.txt below; make sure it exists)
COPY requirements.txt /app/requirements.txt

# Install system dependencies (if necessary). Using --no-cache-dir to keep image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy application code (models and python files)
# Ensure iris_model.pkl and iris_target_names.pkl are in the same directory when building,
# or you can mount at runtime using docker run -v.
COPY . /app

# Expose port that the app will run on
EXPOSE 8000

# Default command to run the app using uvicorn.
# It binds to 0.0.0.0 so the container accepts external connections.
# --proxy-headers is helpful if you run behind a reverse proxy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
