import pandas as pd

# Load with legacy pickle protocol
data = pd.read_pickle("data/LSWMD.pkl")

# Re-save with the latest pickle protocol
data.to_pickle("data/LSWMD_new.pkl", protocol=5)
