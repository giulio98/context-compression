# context-compression

<p align="center">
    <a href="https://github.com/giulio98/context-compression/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/giulio98/context-compression/Test%20Suite/main?label=main%20checks></a>
    <a href="https://giulio98.github.io/context-compression"><img alt="Docs" src=https://img.shields.io/github/deployments/giulio98/context_compression/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A new awesome project.


## Installation

```bash
pip install git+ssh://git@github.com/giulio98/context-compression.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:giulio98/context-compression.git
cd context-compression
pre-commit install
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t image-context-compression -f docker/Dockerfile .
nvidia-docker run --detach -v /path/to/context-compression:/home/jovyan/context-compression image-context-compression tail -f /dev/null
```

Then exec the container
```bash
nvidia-docker exec -it container_id /bin/bash
```

Run the scripts
```
tmux
sh scripts/finetune_squad_v2_qa.sh
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
