import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import selection_rate, false_positive_rate, true_positive_rate
from downstream_fairness.process import get_adjusted_scores
from typing import Dict
import warnings
warnings.filterwarnings("ignore")


def _create_results(dataset: pd.DataFrame,
                    lam_dict: Dict[str, float],
                    adjustment_table: pd.DataFrame,
                    sens_col: str,
                    label_col: str,
                    score_col: str,
                    thresholds: np.ndarray = np.linspace(0, 1, 101)) -> pd.DataFrame:
    """
    This function creates a table for visualizing different metrics across threshold values.

    :param dataset: The dataset you want to visualize adjusted scores
    :param lam_dict: Dictionary of metric, lambda value pairs. Generated from get_lambdas in train.py
    :param adjustment_table: DataFrame containing adjustments generated from get_adjustment_table in train.py
    :param sens_col: The column name that contains the sensitive attribute values in dataset
    :param label_col: The column name that contains the labels in dataset
    :param score_col: The column name that contains the scores in dataset
    :param thresholds: A list of threshold values, which is defaulted to the values 0 to 1, incremented by .01

    :return: DataFrame containing the results for each lambda by each metric to visualize
    """

    resultdf = pd.DataFrame({})
    groups = dataset[sens_col].values
    groups_set = list(set(dataset[sens_col].values))
    scores = dataset[score_col].values
    labels = dataset[label_col].values

    # Adding overall accuracy and per group accuracy
    # Overall accuracy should showcase how accuracy changes minimally for each lambda value
    metrics = list(lam_dict.keys())
    metrics.append('accuracy_overall')
    metrics.append('accuracy')
    metrics[metrics.index('eqodds')] = 'selection_rate'

    # Adding 0 to show the original prediction probabilities
    # Adding 1 to show strong demographic parity, otherwise known as a full adjustment
    lam_vals = list(lam_dict.values())
    lam_vals.append(0.0)
    lam_vals.append(1.0)

    for metric in metrics:
        for lam in lam_vals:
            scores = get_adjusted_scores(adjustment_table, dataset, sens_col, score_col, lam=lam)
            if metric == 'accuracy_overall':
                metric_list = []
                for threshold in thresholds:
                    preds = scores > threshold
                    metric_list.append(np.mean(labels == preds))
                resultdf[f'{metric}_lam_{lam}_overall'] = metric_list
            else:
                for group in groups_set:
                    metric_list = []
                    for threshold in thresholds:
                        preds = scores > threshold
                        if metric == 'tpr':
                            metric_list.append(true_positive_rate(labels[groups == group], preds[groups == group]))
                        elif metric == 'fpr':
                            metric_list.append(false_positive_rate(labels[groups == group], preds[groups == group]))
                        elif metric == 'accuracy':
                            metric_list.append(np.mean(labels[groups == group] == preds[groups == group]))
                        elif metric == 'selection_rate':
                            metric_list.append(selection_rate(labels[groups == group], preds[groups == group]))
                    resultdf[f'{metric}_lam_{lam}_{group}'] = metric_list

    return resultdf, lam_vals, metrics, groups_set


def visualize_all_results(dataset: pd.DataFrame,
                          lam_dict: Dict[str, float],
                          adjustment_table: pd.DataFrame,
                          sens_col: str,
                          label_col: str,
                          score_col: str,
                          thresholds: np.ndarray = np.linspace(0, 1, 101)):
    """
    This function is meant to visualize the results generated by _create_results. This will show a shared plot of
    all the metrics for every lambda, including no adjustment and full adjustment.

    :param dataset: The dataset you want to visualize adjusted scores
    :param lam_dict: Dictionary of metric, lambda value pairs. Generated from get_lambdas in train.py
    :param adjustment_table: DataFrame containing adjustments generated from get_adjustment_table in train.py
    :param sens_col: The column name that contains the sensitive attribute values in dataset
    :param label_col: The column name that contains the labels in dataset
    :param score_col: The column name that contains the scores in dataset
    :param thresholds: A list of threshold values, which is defaulted to the values 0 to 1, incremented by .01
    """
    results_df, lam_vals, metrics, groups_set = _create_results(dataset, lam_dict, adjustment_table, sens_col,
                                                                label_col, score_col,
                                                                thresholds)
    
    nrows, ncols = len(lam_vals), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=True)

    for col_num, metric in enumerate(metrics):
        for row_num, lam in enumerate(lam_vals):
            if metric == 'accuracy_overall':
                sns.lineplot(x=thresholds, y=results_df[f'{metric}_lam_{lam}_overall'], ax=axes[row_num, col_num], label=g)
            else:
                for g in groups_set:
                    sns.lineplot(x=thresholds, y=results_df[f'{metric}_lam_{lam}_{g}'], ax=axes[row_num, col_num], label=g)
            axes[row_num, col_num].legend()
            axes[row_num, col_num].set_xlabel('Threshold')
            axes[row_num, col_num].set_ylabel(metric)
            if metric == 'accuracy_overall':
                axes[row_num, col_num].set_title(metric + f'\nlambda={lam}')
            else:
                axes[row_num, col_num].set_title(metric + f'by group\nlambda={lam}')
                
    plt.tight_layout()
    plt.show()
