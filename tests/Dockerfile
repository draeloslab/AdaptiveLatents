FROM mambaorg/micromamba:2.3.2

WORKDIR /app
COPY --chown=mambauser environment.yml pyproject.toml ./
RUN sed -i "1s/adaptive_latents/base/g" environment.yml && \
    micromamba install --yes --file environment.yml && \
    micromamba clean --all --yes


ARG MAMBA_DOCKERFILE_ACTIVATE=1
# (otherwise python will not be found)
COPY --chown=mambauser adaptive_latents adaptive_latents/
COPY --chown=mambauser tests tests/
COPY --chown=mambauser workspace/demos workspace/demos/
COPY --chown=mambauser workspace/main.py workspace/main.py


RUN pip install -e .


RUN echo "jax_platform_name: 'cpu'" > adaptive_latents_config.yaml


CMD ["python", "-m", "pytest", "."] 


# docker image prune
# docker run -it $(docker build -q . -f tests/Dockerfile)