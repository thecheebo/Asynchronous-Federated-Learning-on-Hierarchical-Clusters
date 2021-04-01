FROM python:3.8

WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

ENV SERVICE_TYPE server
ENV ARG_0 0
ENV ARG_1 0

# command to run on container start
CMD python ./$SERVICE_TYPE.py $ARG_0 $ARG_1
