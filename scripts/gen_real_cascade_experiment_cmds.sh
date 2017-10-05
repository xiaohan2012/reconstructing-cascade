#! /bin/zsh

methods=(closure tbfs no-order)
qs=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
k=10

for method in $methods; do
    for q in $qs; do
	print "python real_cascade_experiment.py  -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/${method}/${q}/"
    done
done
