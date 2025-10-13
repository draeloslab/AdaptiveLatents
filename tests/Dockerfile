FROM mambaorg/micromamba:2.3.2

WORKDIR /app
COPY adaptive_latents adaptive_latents/
COPY tests tests/
COPY environment.yml .
COPY pyproject.toml .

RUN sed -i "1s/adaptive_latents/base/g" environment.yml
# RUN sed -i "1s/adaptive_latents/base/g; /mmica/d; /picard/d;" environment.yml

CMD micromamba install --yes --file environment.yml && \
    micromamba clean --all --yes
# ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# RUN pip install -e .

# CMD ["python", "-m", "pytest", "."] 


# docker image prune
# docker run -it $(docker build -q .)
