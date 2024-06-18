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
        feature_cols (list): List of columns to retain as predictors.
        train_size (float): Proportion (between 0 and 1) of data to ues in the training set. Test set will be the
        remainder. Defaults to 0.8.
        time_series_idx (str): Column name to use to order rows prior to splitting train and test. Defaults to 'epoch'.
        group_col (str): Column name to use for grouping data to ensure equal sampling across subjects. Defaults to
        'ID_day'.
        target_col (str): Column name of variable that is target of model predictions. Default value is 'score'.

    Returns:
        (pd.DataFrame): training data.
        (pd.DataFrame): test data.
    """

    # First enforce correct order of time_col by sorting the values within each group of group_col
    df = df.sort_values(by=[group_col, time_series_idx])

    # Identify the group sizes
    group_sizes = df.groupby(group_col).size()

    # Initialize empty lists to store training and testing indices
    train_indices = []
    test_indices = []

    for group, size in group_sizes.items():
        # Calculate the number of rows for training and testing by group
        n_train_rows = int(train_size * size)

        # Get the indices for training and testing rows
        group_indices = df.index[df[group_col] == group]
        train_indices.extend(group_indices[:n_train_rows])
        test_indices.extend(group_indices[n_train_rows:])

    # Extract training and testing datasets
    train_set = df.loc[train_indices]
    test_set = df.loc[test_indices]

    return train_set, test_set
