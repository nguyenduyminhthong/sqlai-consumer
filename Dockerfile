FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY . .

RUN pip install ./submodules/sqlai-core
