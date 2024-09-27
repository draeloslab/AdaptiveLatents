#!/bin/bash
cd $(dirname "$0")/..

if ! command -v mamba 2>&1 >/dev/null
then
    echo "mamba doesn't exist"
    exit 1
fi

ENV="adaptive_latents"
mamba activate base && mamba remove -y -n $ENV --all
mamba create -y -n $ENV
mamba env update --name $ENV --file environment.yml
conda activate $ENV
pip install -e .
# alternate: mamba env create -y --file environment.yml && mamba activate $ENV
