import numpy as np
import pandas as pd
from downstream_fairness.train import get_adjustment_table, get_lambdas
from downstream_fairness.process import get_adjustments, get_adjusted_scores


def test_get_adjustments_0_1():
    """
    Test to make sure the adjustments are between -1 and 1, so that the scores remain in 0 and 1 inclusive.
    """
    sens_vals = [1, 2, 3, 4, 5] * 20
    sens_vals.append(1)
    binary_labels = [0] * 50 + [1] * 51
    scores = np.linspace(0, 1, 101)
    data = pd.DataFrame(np.array([scores, sens_vals, binary_labels]).T,
                        columns=['scores', 'sensitive_attributes', 'labels'])

    adjustment_table = get_adjustment_table(data, sens_col='sensitive_attributes', score_col='scores')
    lambdas = get_lambdas(adjustment_table,
                          data,
                          sens_col='sensitive_attributes',
                          score_col='scores',
                          label_col='labels')

    for value in lambdas.values():
        adjustments = get_adjustments(adjustment_table,
                                      data,
                                      sens_col='sensitive_attributes',
                                      score_col='scores',
                                      lam=value)

        assert np.all(-1 <= adjustments)
        assert np.all(adjustments <= 1)


def test_get_adjusted_scores_0_1():
    """
    Test to make sure the adjusted scores are between 0 and 1 inclusive.
    """
    sens_vals = [1, 2, 3, 4, 5] * 20
    sens_vals.append(1)
    binary_labels = [0] * 50 + [1] * 51
    scores = np.linspace(0, 1, 101)
    data = pd.DataFrame(np.array([scores, sens_vals, binary_labels]).T,
                        columns=['scores', 'sensitive_attributes', 'labels'])

    adjustment_table = get_adjustment_table(data, sens_col='sensitive_attributes', score_col='scores')
    lambdas = get_lambdas(adjustment_table,
                          data,
                          sens_col='sensitive_attributes',
                          score_col='scores',
                          label_col='labels')

    for value in lambdas.values():
        adjustments = get_adjusted_scores(adjustment_table,
                                          data,
                                          sens_col='sensitive_attributes',
                                          score_col='scores',
                                          lam=value)

        assert np.any(0 <= adjustments)
        assert np.any(adjustments <= 1)
