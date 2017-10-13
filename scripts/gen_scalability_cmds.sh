#! /bin/zsh

graphs=(2-6  2-7  2-8  2-9 2-10  2-11  2-12  2-13  2-14)
for graph in $graphs; do
    print "python paper_experiment.py -g barabasi/${graph}  -m greedy  -l ct -p 0.0  -q 0.1  -o outputs/scalability/${graph}/greedy/qs/0.1.time.pkl --parallel"
done
