# 1. Migrate from HuggingFace to vLLM

## Context

HuggingFace is currently the world's largest pre-trained DL model (PTM)
registry. It offers a number of libraries and tools to access and use hosted
models. These libraries are not optimized (by default) to host and serve PTMs.

vLLM is an inference engine focussed on hosting and serving PTMs with the latest
enhancements and inference optimizations.

## Decision

We will migrate to vLLM. For the time being, we will continue to use Llava 1.6
and Phi. However, these models may change for more performant models in the
future.

## Consequences

We will need to have a script to download the models locally. Additionally, the
server script will have to load the models from the filesystem.
