FROM python:3.12.0-slim

# Instalando o Gunicorn
RUN pip install gunicorn

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando o código do aplicativo para o contêiner
COPY . /app

# Instalando as dependências do aplicativo
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expondo a porta 5000 (porta padrão do Flask)
EXPOSE 5000

# Comando de execução do Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
