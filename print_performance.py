import sys
import pandas as pd
path = sys.argv[1]
df = pd.read_pickle(path)
print(path)
print(df)
print(df.to_latex(float_format='%.2f'))
