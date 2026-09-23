FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so they cache independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# One worker is enough for a classroom tool and keeps memory predictable;
# --proxy-headers makes generated docs work behind the compose nginx.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
