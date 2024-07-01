import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    time_series_idx: str = "epoch",
    group_col: str = "ID_day",
) -> tuple:
    """
    Splits data into train and test sets. Splitting is done equally across all values in the group_col.
    Splits done in time-series fashion - e.g. both train and test are continuous values wrt the time_series_idx.

    Args:
        df (pd.DataFrame):  A pandas dataframe with the data, group var, and some time-series indicator.
        train_size (float): Proportion (between 0 and 1) of data to ues in the training set. Test set will be the
        remainder. Defaults to 0.8.
        time_series_idx (str): Column name to use to order rows prior to splitting train and test. Defaults to 'epoch'.
        group_col (str): Column name to use for grouping data to ensure equal sampling across subjects. Defaults to
        'ID_day'.

    Returns:
        (pd.DataFrame): training data.
        (pd.DataFrame): test data.
    """

    # First enforce correct order of time_col by sorting the values within each group of group_col
    df = df.sort_values(by=[group_col, time_series_idx])

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for group in df[group_col].unique():
        group_df = df.loc[df[group_col] == group]
        total_rows = len(group_df)
        train_rows = int(total_rows * train_size)

        train_set = group_df.iloc[:train_rows]
        test_set = group_df.iloc[train_rows:]

        train_df = pd.concat([train_df, train_set])
        test_df = pd.concat([test_df, test_set])

    return train_df, test_df
