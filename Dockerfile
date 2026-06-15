FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY bayesian_orchestrator /app/bayesian_orchestrator
COPY examples /app/examples
COPY pricing /app/pricing

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["bayesian-orchestrator"]
CMD ["run", "--config", "examples/bayesian-orchestrator/config.yaml"]
