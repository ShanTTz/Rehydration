FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip install --no-cache-dir -e .
COPY . .
CMD ["sh", "scripts/reproduce_revision.sh"]
