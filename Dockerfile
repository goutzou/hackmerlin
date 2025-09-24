# Use the official Playwright image (has Chromium + deps preinstalled)
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

# Create a working directory inside the container
WORKDIR /app

# Copy dependency list and install it
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . /app

# Environment defaults
ENV HEADLESS=true \
    PYTHONUNBUFFERED=1

# Run the agent by default
CMD ["python", "agent_hackmerlin.py"]
