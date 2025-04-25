FROM python:3.9-slim

# Install system dependencies needed by Ryu/Eventlet
RUN apt-get update && apt-get install -y \
    python3-distutils \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default run command - change this if needed
CMD ["ryu-manager", "--observe-links", "ryu_controller.py"]

