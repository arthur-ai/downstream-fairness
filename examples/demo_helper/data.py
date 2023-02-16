import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from folktables import adult_filter, public_coverage_filter, ACSIncome, ACSPublicCoverage, ACSMobility
from aif360.datasets.adult_dataset import AdultDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset


def get_data(dataset: pd.DataFrame, seed: int, verb: bool) -> pd.DataFrame:
    """
    Helper function to get data from specific sources.

    :param dataset: the training dataset
    :param seed: the random seed for the model training
    :param verb: produces printed logs during training

    :raises ValueError: Invalid dataset name

    :return: a pandas DataFrame containing scores from a model trained on the specified dataset. Columns include:
    'score', 'group', 'label'
    """
    if dataset == 'adult_old':
        return gen_adult_probs(seed=seed, verb=verb)
    elif dataset == 'adult_new':
        return gen_new_adult(seed=seed, verb=verb, task='income')
    elif dataset == 'public':
        return gen_new_adult(seed=seed, verb=verb, task='public')
    else:
        raise ValueError('Invalid dataset name')


def gen_adult_probs(seed: int = 0,
                    verb: bool = False,
                    sens: str = 'sex',
                    interv: bool = None,
                    algo: str = 'rf') -> pd.DataFrame:
    """
    Helper function train a model and produce a dataset on the adult dataset.

    :param seed: random seed for the model
    :param verb: used to print logs for training
    :param sens: the sensitive attribute for the dataset
    :param interv: produces an object from aif360
    :param algo: the type of algorithm used to train the model

    :return: a dataframe with scores from a model for the adult dataset. New columns are 'score', 'group', 'label'
    """
    adult = AdultDataset(protected_attribute_names=[sens],
                         privileged_classes=[['Male']],
                         categorical_features=[],
                         features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    # scale
    scaler = MinMaxScaler(copy=False)
    adult.features = scaler.fit_transform(adult.features)

    # return just the scaled and shuffled data if pre/inprocessing
    if interv == 'pre/in':
        return adult.split(1, shuffle=True, seed=seed)[0]  # will eventually be split into 3

    # otherwise split the data and train a classifier (don't drop sensitive feature)
    train, test = adult.split([0.3333333], shuffle=True, seed=seed)  # the "test" portion will eventually be split again

    if algo == 'rf':
        classifier = RandomForestClassifier(random_state=seed).fit(train.features, train.labels.reshape(-1))
    elif algo == 'lr':
        classifier = LogisticRegression(random_state=seed).fit(train.features, train.labels)

    probs = classifier.predict_proba(test.features)

    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:, 1]
    ret['label'] = test.labels
    ret['group'] = test.protected_attributes

    if verb:
        print("trained on:", len(train.labels), "samples")
        print("returning: ", len(ret), "samples")

    return ret


def gen_new_adult(seed: int, verb: bool, task: str = 'income', interv: bool = None) -> pd.DataFrame:
    """
    Helper function train a model and produce a dataset on the new adult dataset.

    :param seed: random seed for the model
    :param verb: used to print logs for training
    :param task: the task for the adult dataset (defined by folktables)
    :param interv: produces an object from aif360

    :return: a dataframe with scores from a model for the new adult dataset. New columns are 'score', 'group', 'label'
    """
    tasks = {
        'income': ACSIncome,
        'public': ACSPublicCoverage,
        'mobility': ACSMobility,
    }
    acs_data = pd.read_csv('data_raw/acs_data.csv')  # data_source.get_data(states=['CA']) #, download=True)

    if verb:
        print("data downloaded")

    if interv is not None:
        if task == 'income':  # this does the work in `df_to_numpy` below
            filtered_df = adult_filter(acs_data)[ACSIncome.features]
        elif task == 'public':
            filtered_df = public_coverage_filter(acs_data)[ACSPublicCoverage.features].apply(
                lambda x: np.nan_to_num(x, -1))
        bld = BinaryLabelDataset(df=filtered_df[:30000], label_names=[tasks[task].target],
                                 protected_attribute_names=[tasks[task].group])
        return bld.split(1, shuffle=True, seed=seed)

    features, label, group = tasks[task].df_to_numpy(acs_data)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        features[:30000], label[:30000], group[:30000], test_size=0.666666667, shuffle=True, random_state=seed)

    if verb:
        print("train test splits made")

    model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=seed))
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:, 1]
    ret['group_multi'] = group_test
    ret['label'] = y_test
    ret['group'] = [1 if gp == 1 else 0 for gp in ret.group_multi]

    if verb:
        print("trained on:", len(X_train), "samples")
        print("returning: ", len(ret), "samples")

    return ret
