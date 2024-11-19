#!/bin/bash

L_values=(14)
a=20
b=40
duplicate_values=($(seq "$a" "$b"))

echo "${duplicate_values[@]}"

start=3.1
end=12
step=0.5

for L in "${L_values[@]}"; do
  W=$start
  while (( $(echo "$W <= $end" | bc -l) )); do
     W=$(echo "$W + $step" | bc -l)
     for dupli in "${duplicate_values[@]}"; do

       echo "Diagonalize system of size L = $L and with randomness W = $W [dupli $dupli]" 
       ./solve_random_heisenberg "$L" "$W" "$dupli" 
    done
  done
done

echo "All simulations completed."
