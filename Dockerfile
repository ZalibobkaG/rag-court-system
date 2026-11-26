FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
# torch из CPU-индекса: в Docker на macOS GPU недоступен, CUDA-сборка не нужна
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
