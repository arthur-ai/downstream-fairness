import numpy as np
import pandas as pd
from .constants import EQUALIZED_ODDS, TPR, FPR, OLD_SCORE, SHIFT


def calculate_group_disparity(group: int,
                              train_df: pd.DataFrame,
                              metric: str,
                              sens_col: str,
                              label_col: str,
                              lam: float = 1.0,
                              abs_value: bool = True,
                              do_shift: bool = True) -> float:
    """
    For each group, this function will calculate the group's disparity for a given metric. Namely, for a given group,
    this will find the difference between the given group's metric value and another group's metric value. Supported
    metrics are true positive rate (tpr), false positive rate (fpr), and equalized odds (eq_odds).

    We provide helper flags, so that if someone wanted to find the raw differences (instead of absolute value) or apply
    a lambda adjustment they can.

    :param group: the group to calculate disparity for
    :param train_df: dataframe containing `score`, `group`, and `label` columns
    :param metric: the metric to use. one of ['fpr', 'tpr', 'eqodds']
    :param sens_col: column name containing the sensitive attribute members
    :param label_col: string value to indicate which column has the labels
    :param lam: lambda value to do an adjustment. Will only apply if do_shift = True.
    :param abs_value: Flag whether to use absolute value in the disparity calculation.
    :param do_shift: Flag whether to do the lambda adjustment or not

    :raises ValueError: Your model must be a binary classifier to use downstream fairness
    :raises ValueError: Lambda must be between 0 and 1, inclusive. Your value is {}.
    :raises ValueError: Provided group {} does not exist in training data
    :raises ValueError: Invalid metric type: {}

    :return: Total disparity for a given group and a given metric
    """
    if len(set(train_df[label_col])) != 2:
        raise ValueError("Your model must be a binary classifier to use downstream fairness.")
    if not 0 <= lam <= 1:
        raise ValueError("Lambda must be between 0 and 1, inclusive. Your value is {}.".format(lam))

    all_groups = pd.unique(train_df[sens_col])
    if group not in all_groups:
        raise ValueError(
            f"Provided group {group} does not exist in training data.")

    disparity = 0.0
    for g in all_groups:
        if g != group:
            if metric == FPR:
                # We use this group ordering because the worst off group is the
                # group with the highest FPR
                disparity += _fpr_diff_fn(lam,
                                          label_col,
                                          train_df[train_df[sens_col] == group],
                                          train_df[train_df[sens_col] == g],
                                          abs_value,
                                          do_shift)
            elif metric == TPR:
                # We use this group ordering because the worst off group is the
                # group with the lowest TPR
                disparity += _tpr_diff_fn(lam,
                                          label_col,
                                          train_df[train_df[sens_col] == g],
                                          train_df[train_df[sens_col] == group],
                                          abs_value,
                                          do_shift)
            elif metric == EQUALIZED_ODDS:
                disparity += _eqodds_diff_fn(lam,
                                             label_col,
                                             train_df[train_df[sens_col] == group],
                                             train_df[train_df[sens_col] == g],
                                             abs_value,
                                             do_shift)
            else:
                raise ValueError(f"Invalid metric type: {metric}")
    return disparity


def opt_objective(
        df: pd.DataFrame,
        worst_off: int,
        metric: str,
        sens_col: str,
        label_col: str):
    """
    A function designed to operate as the objective in the minimize_scalar optimization.

    :param df: table of labeled training data with columns `score`, `group`, `label`
    :param worst_off: the worst off group according to calculate_group_disparity
    :param metric: the metric we are optimizing against
    :param sens_col: where to access the sensitive attributes in the df
    :label_col: the column name to access the labels

    :return: objective as a lambda function to be passed to `minimize_scalar`
    """
    return lambda x: calculate_group_disparity(
        worst_off, df, metric, sens_col, label_col, x, abs_value=True, do_shift=True)


def _tpr_diff_fn(
        x: float,
        label_col: str,
        group_a: pd.DataFrame,
        group_b: pd.DataFrame,
        abs_value: bool = True,
        do_shift: bool = True):
    """
    Helper function to calculate the true positive rate for two selected groups.

    :param x: lambda parameter in a lambda defined function
    :param label_col: string value to indicate which column has the labels
    :param group_a: one of the identified groups
    :param group_b: one of the identified groups
    :param abs_value: whether absolute value should be used in the computation
    :param do_shift: whether to apply the lambda parameter, x, to the computation

    :return: lambda function to calculate the group disparity
    """
    return _uniform_diff_help(x,
                              group_a=group_a[group_a[label_col] == 1],
                              group_b=group_b[group_b[label_col] == 1],
                              absolute_value=abs_value,
                              do_shift=do_shift)


def _fpr_diff_fn(
        x: float,
        label_col: str,
        group_a: pd.DataFrame,
        group_b: pd.DataFrame,
        abs_value: bool = True,
        do_shift: bool = True):
    """
    Helper function to calculate the false positive rate for two selected groups.

    :param x: lambda parameter in a lambda defined function
    :param label_col: string value to indicate which column has the labels
    :param group_a: one of the identified groups
    :param group_b: one of the identified groups
    :param abs_value: whether absolute value should be used in the computation
    :param do_shift: whether to apply the lambda parameter, x, to the computation

    :return: lambda function to calculate the group disparity
    """
    return _uniform_diff_help(x,
                              group_a=group_a[group_a[label_col] == 0],
                              group_b=group_b[group_b[label_col] == 0],
                              absolute_value=abs_value,
                              do_shift=do_shift)


def _eqodds_diff_fn(
        x: float,
        label_col: str,
        group_a: pd.DataFrame,
        group_b: pd.DataFrame,
        abs_value: bool = True,
        do_shift: bool = True) -> float:
    """
    Helper function to calculate equalized odds for two selected groups. Default to weighting the groups equally.
    Upgraded functionality in the future can add different weighting to the groups.

    :param x: lambda parameter in a lambda defined function
    :param label_col: string value to indicate which column has the labels
    :param group_a: one of the identified groups
    :param group_b: one of the identified groups
    :param abs_value: whether absolute value should be used in the computation
    :param do_shift: whether to apply the lambda parameter, x, to the computation

    :return: lambda function to calculate the group disparity
    """
    fpr_diff = _fpr_diff_fn(x, label_col, group_a, group_b, abs_value, do_shift)
    tpr_diff = _tpr_diff_fn(x, label_col, group_b, group_a, abs_value, do_shift)
    metric_fns = [tpr_diff, fpr_diff]
    weights = [0.5, 0.5]
    return np.array([metric_fns[i] * weights[i]
                     for i in range(len(metric_fns))]).sum()


def _uniform_diff_help(lam: float,
                       group_a: pd.DataFrame,
                       group_b: pd.DataFrame,
                       absolute_value: bool = True,
                       do_shift: bool = True) -> float:
    """
    For a given lambda and two groups, this function returns the all-threshold disparity. The specificity of which
    metric is used to calculate the disparity is defined by which of the diff functions are called on. Namely, you can
    call the eq_odds, tpr, or fpr functions, and they will call this function as the primary computation to be done.

    :param lam: lambda value to do the shift. Will only be used if do_shift=True
    :param group_a: one of the identified groups
    :param group_b: one of the identified groups
    :param absolute_value: whether to use the absolute value in the calculation
    :param do_shift: whether to apply the lambda value to shift the probabilities

    :return: the average disparity between two groups
    """
    a_scores = group_a[OLD_SCORE].to_numpy()
    b_scores = group_b[OLD_SCORE].to_numpy()
    a_shift = 0
    b_shift = 0

    if do_shift:
        a_shift = group_a[SHIFT].to_numpy()
        b_shift = group_b[SHIFT].to_numpy()
    a_repair = a_scores + np.multiply(a_shift, lam)
    b_repair = b_scores + np.multiply(b_shift, lam)

    a_pr_rates = _positive_rates(a_repair)
    b_pr_rates = _positive_rates(b_repair)

    if absolute_value:
        avg_abs_diff = np.abs(a_pr_rates - b_pr_rates).mean()
    else:
        avg_abs_diff = np.array(a_pr_rates - b_pr_rates).mean()
    return avg_abs_diff


def _positive_rates(scores: np.ndarray) -> np.ndarray:
    """
    Calculates the positivity rates at each threshold from 0 to 1, given the cdf of the scores distribution.

    :param scores: the scores from the given data

    :return: an array of values, indicating how many entries in the cdf were above each threshold
    """
    cdf = _empirical_cdf(scores)
    return np.array([(cdf >= tau).astype(int).mean()
                     for tau in np.linspace(0, 1, 101)])


def _empirical_cdf(sample: np.ndarray) -> np.ndarray:
    """
    Computes empirical CDF of sample, where sample are the scores for a specific group. If computing for tpr or fpr,
    then the sample is an array of scores of a specific group with a specific binary label, as defined by the metric.

    :param sample: scores for a specific group

    :raises ValueError: Sample score distribution is not a valid CDF

    :return: returns a np.array X s.t. for each i < len(X), X[i] is the CDF value for the score corresponding to bin[i]
    """
    empirical_pdf = np.histogram(
        sample, bins=100, range=(
            0, 1), density=True)[0]
    proposed_cdf = np.cumsum(empirical_pdf / 100)  # CDF-ify the pdf

    if (proposed_cdf[-1] - 1) < 1e-4:  # if proposedCDF is actually a CDF
        proposed_cdf = np.around(proposed_cdf, 2)
        return proposed_cdf
    else:
        raise ValueError("Sample score distribution is not a valid CDF")
