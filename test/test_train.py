import numpy as np
import pandas as pd
from downstream_fairness.train import get_adjustment_table, get_lambdas

def test_get_adjustments_table_size():
    """
    Test to make sure the table that is outputted by get_adjustments_table is the number of groups + 2.
    """
    sens_vals = [1, 2, 3, 4, 5] * 20
    sens_vals.append(1)
    scores = np.linspace(0, 1, 101)
    data = pd.DataFrame(np.array([scores, sens_vals]).T, columns=['scores', 'sensitive_attributes'])

    adjustment_table = get_adjustment_table(data, sens_col='sensitive_attributes', score_col='scores')

    assert len(adjustment_table.columns) == 2*len(set(data['sensitive_attributes'])) + 1


def test_get_lambdas_0_1():
    """
    Test to make sure lambda values are between 0 and 1.
    """
    sens_vals = [1, 2, 3, 4, 5] * 20
    sens_vals.append(1)
    binary_labels = [0]*50 + [1]*51
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
        assert 0 <= value <= 1


