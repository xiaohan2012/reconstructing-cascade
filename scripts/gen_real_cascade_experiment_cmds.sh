#! /bin/zsh

methods=(closure tbfs no-order)
qs=(0.001 0.002 0.004 0.008 0.016 0.032)
k=8
cascade_ids=(9 8 7 6 5 4 3 1 0)

for q in $qs; do  # q first
    for cascade_id in $cascade_ids; do
	for method in $methods; do	
	    print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/cascade_${cascade_id}"
	done
    done
done
