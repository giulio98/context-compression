FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN  mkdir -p /home/jovyan/context-compression

WORKDIR /home/jovyan/context-compression

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/context-compression"
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y \
        git \
        vim \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy necessary files and directories
COPY . .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .[dev]

USER root
RUN pip uninstall -y transformer-engine
RUN groupadd -g $GROUP_ID jovyan && \
    useradd -u $USER_ID -g jovyan -m jovyan && \
    usermod -aG video jovyan && \
    chown -R jovyan:jovyan /home/jovyan
USER jovyan

CMD ["/bin/sh"]
