import pandas as pd
import numpy as np

df = pd.read_csv("dataset/zinc_standard_agent/raw/zinc15_250K_2D.csv", sep=',')

# First attempt, if mwt < 500 and logp < 5, it is soluble
df['ZINC_soluble'] = np.where((df['mwt'] < 500) & (df['logp'] < 5), 1, -1)

# Second attempt, if reactive equals to zero
df['ZINC_reactive'] = np.where(df['reactive'] == 0, 1, -1)

print(df.shape)
#print(df['ZINC_soluble'].value_counts())
print(df['ZINC_reactive'].value_counts())

df.to_csv("zinc.csv", index=False)