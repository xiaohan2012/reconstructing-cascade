#! /bin/zsh

methods=(greedy tbfs no-order closure)
# methods=(closure)
# qs=(0.001 0.002 0.004 0.008 0.016 0.032 0.064)
qs=(0.1 0.2 0.3 0.4 0.5)
k=48
cascade_ids=({0..9})
# cascade_ids=({0..9})
# suffix="--evaluate True"
suffix="--small_cascade "

for q in $qs; do  # q first
    for cascade_id in $cascade_ids; do
	for method in $methods; do
	    check_path="outputs/real_cascade_experiment/small_cascade_${cascade_id}/${method}/${q}/7.pkl"
	    if [[ ! -a ${check_path} ]]; then
		print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/small_cascade_${cascade_id} ${suffix}"
	    else	
		print "python real_cascade_experiment.py  -i ${cascade_id} -m ${method}  -q ${q} -k ${k} -o outputs/real_cascade_experiment/small_cascade_${cascade_id} ${suffix}"	
	    fi
	done
    done
done
