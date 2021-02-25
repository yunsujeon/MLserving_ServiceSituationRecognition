#FROM ubuntu:16.04
FROM ubuntu:18.04

#20.04에선 기본 3.8이 제공되기때문에 3.6깔아도 묻히지
#기본 3.6위해 18.04 하자

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3.6.2 \
    python3-pip \
    libgtk2.0-dev
    
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
