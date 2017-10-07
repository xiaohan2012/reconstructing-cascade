#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
# gtypes=(p2p-gnutella08)
# gtypes=(arxiv-hep-th)
gtypes=(grqc)
# gtypes=(email-eu)
# gtypes=(arxiv-hep-th)
# gtypes=(facebook)

methods=(closure tbfs no-order)

model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.2192180745371594" "sp")  # grqc
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.03842231539791447" "sp")  # email-eu
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.32221845177027353" "sp")  # arxiv
# model_params=("si -p 0.5" "ct -p 0.0" "ic -p 0.3079311820652021" "sp")  # facebook

k=100
qs=(0.0010 0.0020 0.0040 0.0080 0.0160 0.0320 0.0640 0.1280)
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
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.pkl ${misc}"
		    # print done
		fi
		
	    done
	done
    done
done
