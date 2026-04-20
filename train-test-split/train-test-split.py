#%%
import pandas as pd
from sklearn.model_selection import train_test_split
# %%

df = pd.read_csv("../data/500hits.csv", encoding='latin-1')
df.head()

# %%

X = df.drop(columns=["PLAYER", "HOF"]) # features
y = df["HOF"] # target

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# %%
