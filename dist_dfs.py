from typing import List
import numpy as np
import pandas as pd


def _first_term(
    A: np.ndarray,
    dim1: int,
    dim2: int,
) -> np.ndarray:
    return (A**2).sum(axis=1).reshape(dim1, 1) * np.ones(shape=(dim1, dim2))


def _second_term(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    return -2 * np.matmul(A, B.T)


def _third_term(
    A: np.ndarray,
    dim1: int,
    dim2: int,
) -> np.ndarray:
    return ((A**2).sum(axis=1).reshape(dim2, 1) * np.ones(shape=(dim2, dim1))).T


def _dist_dfs(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    cols: List[str],
) -> np.ndarray:
    left_len, _ = df_left.shape
    right_len, _ = df_right.shape
    col_len = len(cols)

    left_mat = df_left[cols].to_numpy()
    right_mat = df_right[cols].to_numpy()

    first_term = _first_term(
        A=left_mat,
        dim1=left_len,
        dim2=right_len,
    )

    second_term = _second_term(
        A=left_mat,
        B=right_mat,
    )

    third_term = _third_term(
        A=right_mat,
        dim1=left_len,
        dim2=right_len,
    )

    return first_term + second_term + third_term


def dist_dfs(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    scores = _dist_dfs(df_left=df_left, df_right=df_right, cols=cols)
    scores_df = pd.DataFrame(scores)
    return scores_df


def dist_dfs_chunk(
    df_left: pd.DataFrame, df_right: pd.DataFrame, cols: List[str], chunk_size: int
) -> pd.DataFrame:
    scores = []
    for chunk in np.array_split(df_left, chunk_size):
        scores.append(
            _dist_dfs(
                df_left=chunk,
                df_right=df_right,
                cols=cols,
            )
        )

    scores_df = pd.DataFrame(np.concatenate(scores))
    return scores_df
