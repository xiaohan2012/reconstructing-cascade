#! /bin/zsh

methods=(closure tbfs no-order)
qs=(0.001 0.002 0.004 0.008 0.016 0.032 0.064)
k=8
cascade_ids=({1..10})

for q in $qs; do  # q first
    for cascade_id in $cascade_ids; do
	for method in $methods; do
	    check_path="outputs/real_cascade_experiment/cascade_${cascade_id}/${method}/${q}/7.pkl"
	    if [[ ! -a ${check_path} ]]; then
		print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/cascade_${cascade_id}"
	    else
		# print "${check_path} exists"
	    fi
	done
    done
done
