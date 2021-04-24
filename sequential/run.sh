#!/bin/bash

ROUNDS=5
lr=0.001
l2_lambda=0.01
select_rate=0.9

for N_CLIENTS in 10 20; do
	for per in 2 5; do
		N_LEADERS=$((N_CLIENTS/per))
		for beta in 8; do
			echo "===> python main.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS"
			python main.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS
		done
	done
done

