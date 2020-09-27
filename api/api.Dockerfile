FROM python:3.7

# set the working directory in the container
WORKDIR /srv

# copy the dependencies file to the working directory
COPY api/requirements.txt ./

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY api ./api
COPY ./bin/download_models_api.sh .
RUN ./download_models_api.sh

# command to run on container start
CMD gunicorn -w 4 -b 0.0.0.0:8000 api.server:application --reload --timeout 120
