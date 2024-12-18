FROM python:3.10-slim

# Set up environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install git (if needed) and any other system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create a directory for the app
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script
COPY inference.py inference.py

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
