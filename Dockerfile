FROM python:3.7-slim

MAINTAINER mikulas.bankovic27@gmail.com

RUN apt-get update && apt-get install -y sudo git
RUN apt
RUN ln -fs /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip==20.0.2 setuptools wheel


COPY requirements.txt /implementation-system/

WORKDIR /implementation-system
RUN pip install -r requirements.txt

ADD . /implementation-system

RUN python setup.py develop