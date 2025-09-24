# utils/id_split.py
import pandas as pd

def load_tall(csv_path):
    df = pd.read_csv(csv_path)
    groups = []
    for gid, g in df.groupby("ID", sort=True):
        if len(g) == 200:
            a = float(g["Label_1(Acetal)"].iloc[0])
            b = float(g["Label_2(Air)"].iloc[0])
            groups.append({"ID": int(gid), "Acetal": a, "Air": b})
    return pd.DataFrame(groups)

def filter_csv_by_ids(csv_path, id_list):
    """Return a new DataFrame containing only rows for IDs in id_list."""
    df = pd.read_csv(csv_path)
    id_set = set(map(int, id_list))
    return df[df["ID"].isin(id_set)].copy()
