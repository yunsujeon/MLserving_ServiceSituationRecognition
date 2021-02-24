#FROM ubuntu:16.04
FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3.6.2 \
    python3-pip
# RUN pip3 install --upgrade pip==9.0.3 \
RUN pip3 install --upgrade pip==21.0.1 \
    && pip3 install setuptools

# for flask web server
EXPOSE 8888

# set working directory
ADD . /app
WORKDIR /app

# install required libraries
RUN pip3 install -r requirements.txt

# This is the runtime command for the container
CMD python3 app.py
