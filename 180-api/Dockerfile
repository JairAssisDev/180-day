FROM python:3.11.6-slim-bookworm
LABEL authors="jairvictor"

RUN pip install gunicorn

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
