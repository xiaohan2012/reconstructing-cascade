#! /bin/zsh

# methods=(closure tbfs no-order)
methods=(closure tbfs no-order greedy)
# qs=(0.001 0.002 0.004 0.008 0.016 0.032 0.064)
qs=(0.002 0.004 0.008 0.016 0.032)
k=8
cascade_ids=({5..18})
# cascade_ids=({0..9})
suffix="--evaluate True"
# suffix=""

for q in $qs; do  # q first
    for cascade_id in $cascade_ids; do
	for method in $methods; do
	    # check_path="outputs/real_cascade_experiment/cascade_${cascade_id}/${method}/${q}/.pkl"
	    # if [[ ! -a ${check_path} ]]; then
		# print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/cascade_${cascade_id} ${suffix}"
	    # else	
	    print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/cascade_${cascade_id} ${suffix}"	
	    # fi
	done
    done
done
