FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    prefect \
    mlflow \
    fastapi \
    uvicorn

# Copy application files
COPY flow.py /app/flow.py
COPY SSE_PT10kemb.pth /app/SSE_PT10kemb.pth
COPY utilities.py /app/utilities.py

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
ENTRYPOINT ["python", "flow.py"]
