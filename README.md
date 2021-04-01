

# federated-learning project

## baseline run:     
###	python server.py 0 <num_of_clients>         
###	python client.py 0 <num_of_clients>     
	
## With leaders run:
###	python server.py <num_of_leaders> 0        
###	python leader.py <num_of_leaders> <num_of_clients> 
###	python client.py <num_of_leaders> <num_of_clients> 

## Build with Docker-compose:
### Build/Run:
```shell
docker-compose build # build all three types of services
docker-compose up # run all three types of services 
```

### Modify Number of clients/servers
Edit `docker-compose.yaml` to change `ARG_0` and `ARG_1` for each service type.
For details see above section commands.


Major files:     
	server.py,      
	leader.py,      
	client.py,      
	devices.py

notes:
	https://heartbeat.fritz.ai/federated-learning-demo-in-python-training-models-using-federated-learning-part-3-73cf04cfda32      
	https://github.com/ahmedfgad/FederatedLearning/blob/master/server.py
