import pandas as pd
import glob

paths = glob.glob('./datasets/*.csv')

df = pd.DataFrame()
for path in paths:
    mse_df = pd.read_csv(path, index_col=0)
    df = pd.concat([mse_df, df])

df = round(df * 100000, 2)
df.sort_values(by='Average', axis=0, inplace=True)
print(df)
print('전체 수', len(df))
# df.to_csv('./datasets/mse_total_83.csv', index=True)
print('mse 최솟값 상위 20개' , df.index[:20])