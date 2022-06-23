FROM debian:stable

RUN export DEBIAN_FRONTEND="noninteractive" && \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get install --no-install-recommends -y \
apt-utils \
build-essential \
git \
libdbus-glib-1-2 \
libffi-dev \
libgtk-3-0 \
libx11-xcb1 \
libxt6 \
nodejs \
npm \
pkgconf \
python3-dev \
python3-pip \
wget \
xvfb \
&& \
apt-get clean -y

COPY miniconda.sha256 /

ENV CONDA_DIR /opt/conda

RUN cd / && \
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh && \
sha256sum -c miniconda.sha256 && \
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --set always_yes yes --set changeps1 no && \
conda update -q conda && \
conda info -a

COPY code/requirements.txt /requirements.txt

RUN cd / && \
python3 -m pip install --upgrade pip && \
python3 -m pip install --upgrade wheel && \
python3 -m pip install 'numpy==1.19.5' && \
python3 -m pip install -r requirements.txt

RUN cd / && \
git clone https://github.com/sandrasiby/OpenWPM.git && \
cd /OpenWPM && \
git checkout webgraph && \
bash install.sh && \
mv firefox-bin /opt/firefox-bin

ENV FIREFOX_BINARY /opt/firefox-bin/firefox-bin

RUN mkdir /WebGraph

COPY code /WebGraph/code
COPY robustness /WebGraph/robustness

ENTRYPOINT ["/bin/bash"]
