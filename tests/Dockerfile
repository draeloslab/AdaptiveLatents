FROM mambaorg/micromamba:2.3.2

WORKDIR /app
COPY --chown=mambauser pyproject.toml ./

ARG MAMBA_DOCKERFILE_ACTIVATE=1
run micromamba install -n base -c conda-forge python=3.11 git gcc

COPY --chown=mambauser adaptive_latents adaptive_latents/
COPY --chown=mambauser tests tests/
COPY --chown=mambauser workspace/demos workspace/demos/
COPY --chown=mambauser workspace/main.py workspace/main.py

RUN pip install -e .[dev]

RUN echo "jax_platform_name: 'cpu'" > adaptive_latents_config.yaml


CMD ["python", "-m", "pytest", "."]


# docker image prune
# docker run -it $(docker build -q . -f tests/Dockerfile)