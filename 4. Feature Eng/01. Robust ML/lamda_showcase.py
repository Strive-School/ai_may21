import pandas as pd


df = pd.DataFrame( {'num':[1,2,3,4,5]} )


f = lambda damian: damian +1 


df = df['num'].map(f)

print(df)