FROM python:3.7

ENV PYTHONUNBUFFERED 1

# WORKDIR /tmp

# # needed for add-apt-repository
# RUN apt-get update && apt-get install -y software-properties-common

# # https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# RUN apt-get update
# RUN apt-get install -y cuda-10-1

# set the working directory in the container
WORKDIR /srv

# copy the dependencies file to the working directory
COPY api/requirements.txt ./

RUN pip install -r requirements.txt

# install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY api ./api
# COPY ./bin/download_models_api.sh .
# RUN ./download_models_api.sh

# command to run on container start
CMD gunicorn -w 4 -b 0.0.0.0:8000 api.server:application --reload --timeout 240
