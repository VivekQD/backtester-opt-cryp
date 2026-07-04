import pandas as pd

data=pd.read_parquet("btc_usdc_last7d.parquet")
d=data.head(100)
print(d)