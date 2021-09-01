
# Asynchronous Federated Learning on Hierarchical Clusters Research Paper

In AsyncHierFed Learning, the central server uses either the network topology or some clustering algorithm to assign clusters for workers (i.e., client
devices). In each cluster, a special aggregator device is selected to enable hierarchical learning, leads to efficient communication between server and workers, so that the
burden of the server can be significantly reduced. In addition, asynchronous federated learning schema is used to tolerate heterogeneity of the system 

##### Table of Contents  

TBD 

[Setup](#setup)  

## Setup

## baseline run:     
###	python server.py 0 <num_of_clients>         
###	python client.py 0 <num_of_clients>     
	
## With leaders run:
###	python server.py <num_of_leaders> 0        
###	python leader.py <num_of_leaders> <num_of_clients> 
###	python client.py <num_of_leaders> <num_of_clients> 

Major files:     
	server.py,      
	leader.py,      
	client.py,      
	devices.py

notes:
	https://heartbeat.fritz.ai/federated-learning-demo-in-python-training-models-using-federated-learning-part-3-73cf04cfda32      
	https://github.com/ahmedfgad/FederatedLearning/blob/master/server.py
