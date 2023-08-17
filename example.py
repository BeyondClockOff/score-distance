import numpy as np
import pandas as pd
from dist_dfs import dist_dfs, dist_dfs_chunk


user_dim = 100000
A = pd.DataFrame(
    {
        "user_num": [str(_) for _ in range(user_dim)],
        "score1": np.random.normal(0, 1, user_dim),
        "score2": np.random.normal(0, 1, user_dim),
        "score3": np.random.normal(0, 1, user_dim),
    }
)

product_dim = 100
B = pd.DataFrame(
    {
        "product_num": [str(_) for _ in range(product_dim)],
        "score1": np.random.normal(0, 1, product_dim),
        "score2": np.random.normal(0, 1, product_dim),
        "score3": np.random.normal(0, 1, product_dim),
    }
)


cols = ["score1", "score2", "score3"]
scores = dist_dfs(A, B, cols)
scores = dist_dfs_chunk(A, B, cols, 1000)
