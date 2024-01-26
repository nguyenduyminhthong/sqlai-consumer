FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./submodules/sqlai-core
