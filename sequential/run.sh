#!/bin/bash

ROUNDS=300
lr=0.001
l2_lambda=0.01
select_rate=0.9

for seed in 1 2 3 4 5 10 20 55 66 88 99 100; do
    for N_CLIENTS in 20; do
    	for per in 5; do
    		N_LEADERS=$((N_CLIENTS/per))
    		for beta in 8; do
    			echo "===> python main.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed"
    			python main.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed
    		done
    	done
    done
done

