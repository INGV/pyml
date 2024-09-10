FROM debian:bullseye-slim

LABEL maintainer="Raffaele Di Stefano <raffaele.distefano@ingv.it>"
ENV DEBIAN_FRONTEND=noninteractive

# Installing all needed applications and dependencies for pyrocko
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    gfortran \
    build-essential \
    libglib2.0-0 \
    libglib2.0-dev \
    libfftw3-dev \
    libsqlite3-dev \
    libproj-dev \
    libgeos-dev \
    libnetcdf-dev \
    liblapack-dev \
    libatlas-base-dev \
    systemd \
    wget \
    zip \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for numpy
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Adding python3 libraries
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir obspy
RUN pip3 install --no-cache-dir pyrocko
RUN pip3 install --no-cache-dir plotly sklearn scipy pandas geographiclib

COPY ./dbona_magnitudes_stations_corrections_extended_mq.csv /opt/dbona_magnitudes_stations_corrections_extended_mq.csv
COPY ./pyml.py /opt/pyml.py
COPY entrypoint.sh /opt

WORKDIR /opt
ENTRYPOINT ["bash", "/opt/entrypoint.sh"]
