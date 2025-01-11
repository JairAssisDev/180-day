FROM python:3.11.6-slim-bookworm
LABEL authors="jairvictor"

RUN pip install uvicorn

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]