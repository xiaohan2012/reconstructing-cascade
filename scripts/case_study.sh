#! /bin/zsh

for i in {0..9}; do
    python3 real_cascade_experiment.py  -i ${i} -m closure  -q 0.2 -k 1 -o outputs/real_cascade_experiment/small_cascade_${i} --small_cascade
    python3 real_cascade_experiment.py  -i ${i} -m closure  -q 0.2 -k 1 -o outputs/real_cascade_experiment/small_cascade_${i} --small_cascade --evaluate True
done
