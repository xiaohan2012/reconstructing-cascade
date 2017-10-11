#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
# gtypes=(arxiv-hep-th enron-email)
# gtypes=(arxiv-hep-th facebook grqc email-eu)
gtypes=(grqc email-eu arxiv-hep-th facebook)

methods=(tbfs closure no-order)

models=('si' 'ic' 'ct' 'sp' )

first="${gtypes[1]}"

qs=(0.001 0.002 0.004 0.008 0.016 0.032 0.064 0.128)

for gtype in $gtypes; do
    for method in $methods; do
	for model in $models; do
	    model=("${(@s/ /)model}")
	    model=${model[1]}

	    print "python evaluate.py -g ${gtype}  -m ${method}  -l ${model}  -q ${qs}"
	done
    done
done
