# [Choice] Ubuntu version: bionic, focal
ARG VARIANT=focal
FROM mcr.microsoft.com/vscode/devcontainers/base:${VARIANT}

ARG DEBIAN_FRONTEND=noninteractive

# ===
# C++
# ===

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        colordiff \
        cppcheck \
        cmake \
        cmake-curses-gui \
        doxygen \
        gdb \
        graphviz \
        iputils-ping \
        ninja-build \
        tree \
        valgrind \
    &&\
    rm -rf /var/lib/apt/lists/*

# ======
# Python
# ======

# Setup Python
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        virtualenv \
    &&\
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:$PATH
ARG GET_PYTHON_VERSION="python3 -c 'from sys import version_info as i; print(f\"{i[0]}.{i[1]}\")'"

USER vscode
RUN sudo mkdir -p ${VIRTUAL_ENV} &&\
    sudo chown vscode:vscode ${VIRTUAL_ENV} &&\
    virtualenv -p python$(eval ${GET_PYTHON_VERSION}) ${VIRTUAL_ENV} &&\
    pip install \
        pytest \
        pytest-icdiff \
    &&\
    rm -rf $HOME/.cache/pip
USER root

# ============
# gym-ignition
# ============

# iDynTree
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        python3-numpy libxml2-dev coinor-libipopt-dev libeigen3-dev libassimp-dev swig &&\
    rm -rf /var/lib/apt/lists/*
USER vscode
RUN pip install git+https://github.com/robotology/idyntree@devel &&\
    rm -rf $HOME/.cache/pip
ENV CMAKE_PREFIX_PATH=${VIRTUAL_ENV}/lib/python3.8/site-packages/idyntree:${CMAKE_PREFIX_PATH}
USER root

# Ignition
ARG IGNITION_DISTRIBUTION="dome"
ARG IGNITION_DEFAULT_CHANNEL="stable"
RUN echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-${IGNITION_DEFAULT_CHANNEL} `lsb_release -cs` main" > \
        /etc/apt/sources.list.d/gazebo-${IGNITION_DEFAULT_CHANNEL}.list &&\
    wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends ignition-${IGNITION_DISTRIBUTION} &&\
    rm -rf /var/lib/apt/lists/*
