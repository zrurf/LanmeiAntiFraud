FROM python:3.14.8-slim

ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0
# 国内可换成 https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LMAF_HOST=0.0.0.0 \
    LMAF_PORT=8000 \
    LMAF_EXPORT_DIR=/app/artifacts/export/deploy \
    LMAF_WORKERS=2 \
    LMAF_TASK_TTL=3600

RUN pip install --no-cache-dir \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}"

WORKDIR /app

COPY requirements-service.txt ./
RUN pip install --no-cache-dir --index-url "${PIP_INDEX}" -r requirements-service.txt

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin lmaf

COPY --chown=lmaf:lmaf service/ ./service/
COPY --chown=lmaf:lmaf train/ ./train/

COPY --chown=lmaf:lmaf artifacts/export/deploy/ ./artifacts/export/deploy/
COPY --chown=lmaf:lmaf models/macbert/ ./models/macbert/

USER lmaf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json,os,sys,urllib.request;p=os.environ.get('LMAF_PORT','8000');sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:'+p+'/api/v1/health',timeout=8)).get('model_loaded') else 1)"

CMD ["python", "-m", "service.api"]
