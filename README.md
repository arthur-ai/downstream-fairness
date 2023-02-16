### Downstream Fairness

We implement the algorithm presented in [Geometric Repair for Fair Classification at Any Decision Threshold](https://arxiv.org/pdf/2203.07490.pdf).
The algorithm looks at the scores, labels, and sensitive attribute in a dataset, and determines a set of adjustments.
The original set of adjustments allow one to adjust the prediction probabilities of a binary classifier, and perturb
them (by adding the appropriate adjustment to a prediction probability), so that you can achieve `demographic parity`.
To achieve other fairness definitions, such as equalized odds and equal opportunity, our algorithm provides a `lambda`
value. This `lambda` value is multiplied to each entry of the set of adjustments, and then those can be added to
predictions. Here is a good way to think about it:

```
prediction_probability + adjustment_value # achieves demographic parity

prediction_probability + lambda_val * adjustment_value # achieves some fairness definition attached to the lambda value
```

### Quickstart
First, please make sure to download our package. You can either do this from this repo via:

```
pip install -e .
```

or you can install it directly via `pip`:

```
pip install downstream_fairness
```

To get started, we recommend just running our `get_bias_mitigator`, so you can get a dictionary of lambda values and
an adjustment table. To go more in-depth, we recommend going through our demo notebook.

```
from downstream_fairness.process import get_bias_mitigator

# Gets the adjustment table and lambdas
table, lambdas = get_bias_mitigator(YOUR_DATA, 
                                    sens_col=YOUR_SENSITIVE_ATTRIBUTE_COLUMN_NAME, 
                                    score_col=YOUR_PREDICTION_PROBABILTIY_COLUMN, 
                                    label_col=YOUR_LABEL_COLUMN)
                                    
# Adjusts the prediction probabilities on the fly
adjusted_scores = get_adusted_scores(table, 
                                     DATA_YOU_WANT_ADJUSTED, 
                                     YOUR_SENSITIVE_ATTRIBUTE_COLUMN_NAME,
                                     YOUR_PREDICTION_PROBABILTIY_COLUMN,
                                     lambdas[THE_METRIC_YOU_WANT])
```