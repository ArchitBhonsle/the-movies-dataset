import pandas as pd

def get_index(id: int, df: pd.DataFrame):
    found = df[df['id'] == id].index
    if len(found) == 0:
        return None
    return found[0]

def get_id(index: int, df: pd.DataFrame):
    found = df.iloc[index]
    if found.empty:
        return None
    return found['id']