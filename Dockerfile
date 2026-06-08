FROM python:3.12-slim

LABEL maintainer="student@ibu.edu.ba"
LABEL description="News Media Monitoring Pipeline — Dash Dashboard"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-dashboard.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-dashboard.txt

COPY app.py .
COPY src ./src
COPY scripts ./scripts
COPY data/processed/cleaned ./data/processed/cleaned

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120", "app:server"]