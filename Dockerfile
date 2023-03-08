FROM debian:bullseye-slim

LABEL maintainer="Raffaele Di Stefano <raffaele.distefano@ingv.it>"
ENV DEBIAN_FRONTEND=noninteractive

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
RUN python3 -m pip install numpy
RUN python3 -m pip install obspy
RUN python3 -m pip install pyrocko
RUN python3 -m pip install plotly
RUN python3 -m pip install sklearn
RUN python3 -m pip install scipy
RUN python3 -m pip install pandas
RUN python3 -m pip install geographiclib

COPY ./dbona_magnitudes_stations_corrections_extended_mq.csv /opt/dbona_magnitudes_stations_corrections_extended_mq.csv
COPY ./pyml.py /opt/pyml.py
COPY entrypoint.sh /opt

#
WORKDIR /opt
ENTRYPOINT ["bash", "/opt/entrypoint.sh"]
