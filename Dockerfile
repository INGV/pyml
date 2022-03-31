FROM ubuntu:20.04
LABEL maintainer="Raffaele Di Stefano <raffaele.distefano@ingv.it>"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y python3 python3-pip
RUN pip3 install pymysql
RUN pip3 install mysql-connector
RUN pip3 install mysql-connector-python
COPY ./pyml.py /opt/pyml.py
COPY ./pyml.conf /opt/pyml.conf
