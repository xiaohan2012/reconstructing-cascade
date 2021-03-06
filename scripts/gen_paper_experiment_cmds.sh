#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
# gtypes=(p2p-gnutella08)
# gtypes=(arxiv-hep-th)
# gtypes=(grqc)
# gtypes=(email-eu)
gtypes=(arxiv-hep-th)
# gtypes=(facebook)
# gtypes=(grqc email-eu arxiv-hep-th facebook)
# methods=(closure tbfs no-order)
# methods=(greedy)
# methods=(greedy tbfs)
methods=(no-order)
model_params=("ct -p 0.0")

# model_params=("si -p 0.5")
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.2192180745371594" "sp")  # grqc
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.03842231539791447" "sp")  # email-eu
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.32221845177027353" "sp")  # arxiv
# model_params=("si -p 0.5"  "ic -p 0.3079311820652021" "ct")  # facebook

k=100
# qs=(0.001 0.002 0.004 0.008 0.016 0.032 0.064 0.128 0.256)
qs=(0.256)

misc="-k ${k} --parallel"

for gtype in $gtypes; do
    for method in $methods; do
	for model_param in $model_params; do
	    for q in $qs; do
		# temporary
		
		model=("${(@s/ /)model_param}")
		model=${model[1]}

		# add command if not computed
		check_path="outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}/99.pkl"
		if [[ ! -a ${check_path} ]]; then
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.time.pkl ${misc}"
		else
		    # print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.time.pkl ${misc}"
		    # print done
		fi
		
	    done
	done
    done
done
