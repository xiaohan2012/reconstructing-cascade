#! /bin/zsh
# -g grid --param '2-6' \
# -g p2p-gnutella08 --param "" \
# -g pokec --param "" \
# -l ic -p 0.021921807453715934 \
kernprof -l  paper_experiment.py \
  -g grqc \
  -m greedy \
  -l sp \
  -q 0.01 \
  -o outputs/paper_experiment/test.pkl \
  -k 10 \
  -v

