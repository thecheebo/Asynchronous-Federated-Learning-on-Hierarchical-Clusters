#!/bin/bash

if [[ $1 == "a" && $2 == "h" ]]; then
	ROUNDS=10
	lr=0.001
	l2_lambda=0.01
	select_rate=0.9
	
	for seed in 100; do
	    for N_CLIENTS in 20; do
	    	for per in 5; do
	    		N_LEADERS=$((N_CLIENTS/per))
	    		for beta in 8; do
	    			echo "===> python async-hier.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed"
	    			python async-hier.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed
	    		done
	    	done
	    done
	done
elif [[ $1 == "a" && $2 == "n" ]]; then
	ROUNDS=10
	lr=0.001
	l2_lambda=0.01
	select_rate=0.9
	
	for seed in 1; do
	    for N_CLIENTS in 20; do
	        for beta in 8; do
	            echo "===> python async-no-hier.py $N_CLIENTS $lr $l2_lambda $beta $select_rate $ROUNDS $seed"
	            python async-no-hier.py $N_CLIENTS $lr $l2_lambda $beta $select_rate $ROUNDS $seed
	        done
	    done
	done
elif [[ $1 == "s" && $2 == "h" ]]; then
	ROUNDS=500
	lr=0.0001
	l2_lambda=0.01
	select_rate=0.9
	
	for seed in 100; do
	    for N_CLIENTS in 20; do
	        for per in 5; do
	            N_LEADERS=$((N_CLIENTS/per))
	            for beta in 8; do
	                echo "===> python sync-hier.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed"
	                python sync-hier.py $N_CLIENTS $N_LEADERS $lr $l2_lambda $beta $select_rate $ROUNDS $seed
	            done
	        done
	    done
	done
elif [[ $1 == "s" && $2 == "n" ]]; then
	ROUNDS=10
    lr=0.001
    l2_lambda=0.01
    select_rate=0.9

    for seed in 1; do
        for N_CLIENTS in 20; do
            for beta in 8; do
                echo "===> python sync-no-hier.py $N_CLIENTS $lr $l2_lambda $beta $select_rate $ROUNDS $seed"
                python sync-no-hier.py $N_CLIENTS $lr $l2_lambda $beta $select_rate $ROUNDS $seed
            done
        done
    done
else
	echo "Wrong input, thanks."
fi
