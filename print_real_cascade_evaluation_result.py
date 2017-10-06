import os
import pandas as pd

dirname = 'outputs/real_cascade_experiment'
methods = ['closure', 'tbfs', 'no-order']

q = 0.02
measures = ['n.prec', 'n.rec', 'order accuracy']

rows = []
for measure in measures:
    row = []
    for method in methods:
        p = os.path.join(dirname, method, '{}/{}.pkl'.format(q, q))
        df = pd.read_pickle(p)
        row.append(df[measure]['mean'])

    rows.append(row)
    
r = pd.DataFrame(rows, index=measures, columns=methods)
print(r.to_latex())
