FROM debian:bookworm-slim

LABEL maintainer="Raffaele Di Stefano <raffaele.distefano@ingv.it>"
ENV DEBIAN_FRONTEND=noninteractive
# Bookworm marks the system Python as externally managed (PEP 668) and pip
# refuses to install into it. This image is single-purpose, so installing
# system-wide is fine and avoids changing how entrypoint.sh calls python3.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Installing all needed applications
RUN apt-get clean \
    && apt-get update \
    && apt-get dist-upgrade -y --no-install-recommends \
    && apt-get install -y \
        python3 \
        python3-pip \
        gcc \
        build-essential \
        systemd \
        wget \
        zip \
        curl \
        vim

# Adding python3 libraries
RUN python3 -m pip install --upgrade pip
# One resolver run, so the pins hold across all packages: installed one by
# one, a later package silently upgrades numpy past what pyrocko accepts.
# numpy<2 is required by pyrocko on Python <= 3.11; pandas<3 because
# pyml.py assigns float results into columns that pandas 3 no longer
# upcasts (LossySetitemError). Both match what bullseye was resolving to.
RUN python3 -m pip install \
        "numpy<2" \
        obspy \
        pyrocko \
        plotly \
        scipy \
        "pandas<3" \
        geographiclib

COPY ./dbona_magnitudes_stations_corrections_extended_mq.csv /opt/dbona_magnitudes_stations_corrections_extended_mq.csv
COPY ./pyml.py /opt/pyml.py
COPY entrypoint.sh /opt

#
WORKDIR /opt
ENTRYPOINT ["bash", "/opt/entrypoint.sh"]
