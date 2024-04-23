# generative-modeling

<p align="center">
    <a href="https://github.com/giulio98/generative-modeling/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/giulio98/generative-modeling/Test%20Suite/main?label=main%20checks></a>
    <a href="https://giulio98.github.io/generative-modeling"><img alt="Docs" src=https://img.shields.io/github/deployments/giulio98/generative-modeling/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A new awesome project.


## Installation

```bash
pip install git+ssh://git@github.com/giulio98/generative-modeling.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:giulio98/generative-modeling.git
cd generative-modeling
pre-commit install
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t image-generative-modeling -f docker/Dockerfile .
docker run --gpus all --detach -v /path/to/generative-modeling:/home/jovyan/generative-modeling image-generative-modeling tail -f /dev/null
```

Then exec the container
```bash
docker exec -it container_id /bin/bash
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
