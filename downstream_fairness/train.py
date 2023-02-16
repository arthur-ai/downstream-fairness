import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .utils.barycenter import barycenter, transport
from .utils.metrics import calculate_group_disparity, opt_objective
from .utils.constants import (
    REPAIRED_SCORES,
    ORIGINAL_PREDICTIONS,
    ADJUSTED_SCORES,
    SHIFT,
    OLD_SCORE,
    TPR,
    FPR,
    EQUALIZED_ODDS
)
from .process import get_adjustments
from typing import Dict


def get_adjustment_table(data: pd.DataFrame, sens_col: str, score_col: str) -> pd.DataFrame:
    """
    Takes in the reference dataset, and identifications for the sensitive attribute column and the score column.
    This is then used to calculate the barycenter, the value that helps calculate the lambda, via an optimization
    procedure. This barycenter is then used to create an adjustment table that acts as the *full* adjustment to the
    scores to achieve strong demographic parity.

    :param data: the reference dataset formatted with appropriate column names
    :param score_col: a string attribute to call the column that contain scores
    :param sens_col: a string attribute to call the column that contains the sensitive attributes

    :raises ValueError: the column {} does not exist in the provided dataframe.
    :raises ValueError: the column {} does not exist in the provided dataframe.

    :return: a pandas dataframe with the full adjustment table. The adjustment table will have columns 'pred_orig' for
    the original predictions, 'repaired_scores_{group}' for the actual fully adjust scores for each group, and
    'adjusted_scores_{group}' for the adjustment values for each group.
    """
    if sens_col not in data.columns:
        raise ValueError("The column {} does not exist in the provided dataframe.".format(sens_col))
    if score_col not in data.columns:
        raise ValueError("The column {} does not exist in the provided dataframe.".format(score_col))

    formatted_data = data[[sens_col, score_col]]
    sens = formatted_data[sens_col].values
    unique_sens, sens_counts = np.unique(sens, return_counts=True)
    grid = np.linspace(0, 1, 101)
    group_scores = []
    freqs = []
    for sens_attr in unique_sens:
        group_sample = formatted_data[formatted_data[sens_col]
                                      == sens_attr][score_col]
        group_scores.append(group_sample)
        freqs.append(len(group_sample))
    weights = np.array(freqs) / len(formatted_data)
    bc = barycenter(group_scores, weights, grid)
    df_repair = pd.DataFrame(grid, columns=[ORIGINAL_PREDICTIONS])
    for group in unique_sens:
        df_repair[REPAIRED_SCORES + str(group)] = \
            df_repair[ORIGINAL_PREDICTIONS].apply(func=transport,
                                                  args=(formatted_data.loc[formatted_data[sens_col] ==
                                                                           group, score_col].values, bc, grid))

        df_repair[ADJUSTED_SCORES +
                  str(group)] = df_repair[REPAIRED_SCORES + str(group)] - df_repair[ORIGINAL_PREDICTIONS]

    return df_repair


def get_lambdas(adjust_table: pd.DataFrame,
                train_df: pd.DataFrame,
                sens_col: str,
                score_col: str,
                label_col: str) -> Dict[str, float]:
    """
    Utilizing the adjustment table that was generated from the barycenter, we now generate lambda values that correspond
    to various fairness definitions. Technically, this method can generate lambdas for any probablistic, closed form
    fairness definition. Implemented in this version are equalized odds, false positive rate, and true positive rate
    (equal opportunity).

    This function chooses the group with the worst disparity to be the group we are correcting for most. Then, after
    identifying that group, we optimize the lambda value adjustments as to minimize the amount of disparity on this
    group. Each definition outputs its own lambdas values.

    Note, current research is underway to optimize for all groups. For now, this optimization procedure works best for
    binary, but will still achieve significant results for each group, even though it only optimizes for one group.

    :param adjust_table: saved table of adjustments from get_adjustment_table_binary
    :param train_df: table of labeled training data
    :param sens_col: string value to indicate which column have sensitive attributes
    :param score_col: string value to indicate which column has the scores
    :param label_col: string value to indicate which column has the labels

    :raises ValueError: the column {} does not exist in the provided dataframe.
    :raises ValueError: the column {} does not exist in the provided dataframe.
    :raises ValueError: the column {} does not exist in the provided dataframe.
    :raises ValueError: lambda must be between 0 and 1, inclusive. The lambda the optimizer is producing is {}.

    :return: dictionary of lambda values with keys as the corresponding fairness metric (fpr, tpr, eqodds)
    """
    if sens_col not in list(train_df.columns):
        raise ValueError("The column {} does not exist in the provided dataframe.".format(sens_col))
    if score_col not in list(train_df.columns):
        raise ValueError("The column {} does not exist in the provided dataframe.".format(score_col))
    if label_col not in list(train_df.columns):
        raise ValueError("The column {} does not exist in the provided dataframe.".format(label_col))

    _check_binary_class(train_df, sens_col, label_col)

    all_groups = pd.unique(train_df[sens_col])
    metrics = [EQUALIZED_ODDS, FPR, TPR]
    # get the corrected scores for the labeled training data
    train_df[SHIFT] = get_adjustments(adjust_table, train_df, sens_col, score_col)
    # rename column at this point, so we can internally work with the dataframe more easily
    train_df = train_df.rename(columns={score_col: OLD_SCORE}, inplace=False)

    disparities = {}
    worst_off = {}
    lambdas = {}
    for metric in metrics:
        disparities[metric] = [
            (group,
             calculate_group_disparity(
                 group,
                 train_df,
                 metric,
                 sens_col,
                 label_col,
                 abs_value=False,
                 do_shift=False)) for group in all_groups]
        worst_off[metric] = max(disparities[metric], key=lambda x: x[1])

        lambdas[metric] = np.around(
            minimize_scalar(
                opt_objective(
                    df=train_df,
                    worst_off=worst_off[metric][0],
                    metric=metric,
                    sens_col=sens_col,
                    label_col=label_col),
                bounds=(
                    0,
                    1.01),
                method='Bounded').x,
            2)

        if not (0 <= lambdas[metric] <= 1):
            raise ValueError("Lambda must be between 0 and 1, inclusive."
                             "The lambda the optimizer is producing is {}".format(lambdas[metric]))

    return lambdas


def _check_binary_class(data: pd.DataFrame, sens_col: str, label_col: str):
    """
    Checks whether the data provided gives binary classes for the entire dataset and each group has an instance
    with a positive and negative class

    :param data: original dataframe
    :param sens_col: the string that calls the column that contains the sensitive attributes
    :param label_col: the string that calls the column that contains the labels

    :raises ValueError: If the model is not a binary classifier
    :raises ValueError: Each group must have an instance with each binary class label. Group {} does not.
    """
    if len(data[label_col].unique()) != 2:
        raise ValueError("This model must be a binary classifier.")

    groups = list(data[sens_col].unique())

    for group in groups:
        if len(set(data[data[sens_col] == group][label_col])) != 2:
            raise ValueError("Each group must have an instance with each binary class label."
                             "Group {} does not.".format(group))
