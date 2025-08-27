FROM python:3.11-slim

# system deps: libsndfile (soundfile), ffmpeg (任意), fonts (PDF用)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存の先に requirements をコピーするとキャッシュが効く
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . .

# ポートは Spaces が PORT を注入するので EXPOSE は任意
# EXPOSE 7860

# 起動：0.0.0.0:$PORT にバインド（.env は不要）
ENV PYTHONUNBUFFERED=1
CMD ["bash", "-lc", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
