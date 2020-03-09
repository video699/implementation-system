FROM python:3.7

MAINTAINER Mikulas Bankovic

RUN apt-get update && apt-get install -y sudo git

RUN ln -fs /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip==20.0.2 setuptools wheel

COPY . /implementation-system/

WORKDIR /implementation-system

RUN pip install .

RUN pip install jupyter notebook

RUN jupyter notebook --generate-config


ENTRYPOINT ["jupyter", "notebook", "--port=8888", "--ip=*", "--no-browser", "--allow-root"]
