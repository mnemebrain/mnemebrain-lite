FROM python:3.12-slim

RUN pip install --no-cache-dir "mnemebrain-lite[api]"

EXPOSE 8000

CMD ["python", "-m", "mnemebrain_core"]
