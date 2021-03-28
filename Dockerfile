FROM python:3.8

WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
CMD [ "python", "server.py 1" ]
CMD [ "python", "client.py 1" ]

