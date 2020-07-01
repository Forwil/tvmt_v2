FROM ubuntu:16.04
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        make \
        unzip \
        wget

RUN apt-get update && apt-get install -y \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev 

ENV PYENV_ROOT=/opt/pyenv \
    PATH=/opt/pyenv/shims:/opt/pyenv/bin:$PATH \
    CAFFEPATH=/opt/python/caffe

RUN git clone --depth 1 https://github.com/pyenv/pyenv.git /opt/pyenv \
    && eval "$(pyenv init -)" \
    && env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.6.4 \
    && pyenv global 3.6.4

ENV PYTHONPATH=/opt/python:$PYTHONPATH

ENV LD_LIBRARY_PATH=$CAFFEPATH:/usr/local/lib:$LD_LIBRARY_PATH

RUN wget http://10.4.112.29:12345/clang%2Bllvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar -xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    cp clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04/* /usr/local/  -r

RUN apt-get install -y libedit-dev libopenblas-dev

RUN pip install numpy  onnx

RUN pip install --upgrade pip

RUN pip install cmake==3.16.3

RUN ln -sf /opt/pyenv/versions/3.6.4/lib/python3.6/site-packages/cmake/data/bin/cmake /usr/bin

RUN pip install xgboost tornado

RUN mkdir /app

WORKDIR /app

ADD . /app

RUN mkdir build && cd build && cmake .. && make -j4 && cd ..

RUN cd python && python setup.py install 

RUN cd topi/python && python setup.py install

