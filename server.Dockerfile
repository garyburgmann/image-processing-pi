FROM python:3.7

# set the working directory in the container
WORKDIR /srv

# copy the dependencies file to the working directory
COPY requirements.txt server.requirements.txt ./

# install dependencies
RUN pip install -r server.requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-dev

# copy the content of the local src directory to the working directory
COPY server.py ./
COPY app ./app
COPY bin ./bin
COPY labels ./labels
# RUN ./bin/download_models_tflite.sh

# command to run on container start
CMD [ "./bin/start_server.sh" ]
