from time import time
from typing import List

from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from phenotrex.structure.records import TrainingRecord
from phenotrex.util.logging import get_logger
from phenotrex.util.helpers import get_x_y_tn

import numpy as np

DEFAULT_STEP_SIZE = 0.0025
DEFAULT_SCORING_FUNCTION = 'balanced_accuracy'


def compress_vocabulary(records: List[TrainingRecord], pipeline: Pipeline):
    """
    Method to group features, that store redundant information,
    to avoid overfitting and speed up process (in some cases).
    Might be replaced or complemented by a feature selection method in future versions.

    :param records: a list of TrainingRecord objects.
    :param pipeline: the targeted pipeline where the vocabulary should be modified
    :return: nothing, sets the vocabulary for CountVectorizer step
    """
    X, y, tn = get_x_y_tn(records)  # we actually only need X
    vec = pipeline.named_steps["vec"]
    if not vec.vocabulary:
        vec.fit(X)
        names = [name for name, i in vec.get_feature_names()]
    else:
        names = sorted(vec.vocabulary, key=vec.vocabulary.get)

    X_trans = vec.transform(X)

    seen = {}
    new_vocabulary = {}
    new_index = 0
    for i in range(len(names)):
        column = X_trans.getcol(i).nonzero()[0]
        key = tuple(column)
        found_id = seen.get(key)
        if not found_id:
            seen[key] = new_index
            new_vocabulary[names[i]] = new_index
            new_index += 1
        else:
            new_vocabulary[names[i]] = found_id

    # set vocabulary to vectorizer
    pipeline.named_steps["vec"].vocabulary = new_vocabulary
    pipeline.named_steps["vec"].vocabulary_ = new_vocabulary
    pipeline.named_steps["vec"].fixed_vocabulary_ = True


def recursive_feature_elimination(records: List[TrainingRecord], pipeline: Pipeline,
                                  step: float = DEFAULT_STEP_SIZE,
                                  n_features: int = None,
                                  random_state: np.random.RandomState = None):
    """
    Function to apply RFE to limit the vocabulary used by the CustomVectorizer, optional step.

    :param records: list of TrainingRecords, entire training set.
    :param pipeline: the pipeline which vocabulary should be modified
    :param step: rate of features to eliminate at each step. the lower the number, the more steps
    :param n_features: number of features to select (if None: half of the provided features)
    :param random_state: random state for deterministic results
    :return: number of features used
    """

    # TODO: enable logging (optional)

    t1 = time()

    X, y, tn = get_x_y_tn(records)
    vec = pipeline.named_steps["vec"]
    estimator = pipeline.named_steps["clf"]

    # get previous vocabulary (might be already compressed)
    if not vec.vocabulary:
        vec.fit(X)
        previous_vocabulary = {name: i for name, i in vec.get_feature_names()}
    else:
        previous_vocabulary = vec.vocabulary

    if not n_features:
        n_features = len(previous_vocabulary) // 2

    X_trans = vec.transform(X)

    logger = get_logger(__name__, verb=True)
    split = StratifiedKFold(shuffle=True, n_splits=5, random_state=random_state)
    selector = RFECV(estimator, step=step, min_features_to_select=n_features, cv=split, n_jobs=5,
                     scoring=DEFAULT_SCORING_FUNCTION)
    selector = selector.fit(X=X_trans, y=y)

    original_size = len(previous_vocabulary)
    support = selector.get_support()
    support = support.nonzero()[0]
    new_id = {support[x]: x for x in range(len(support))}
    vocabulary = {feature: new_id[i] for feature, i in previous_vocabulary.items() if
                  not new_id.get(i) is None}
    size_after = selector.n_features_

    t2 = time()

    logger.info(
        f"{size_after} features were selected of {original_size} using Recursive Feature Eliminiation"
        f" in {np.round(t2 - t1, 2)} seconds.")

    # set vocabulary to vectorizer
    pipeline.named_steps["vec"].vocabulary = vocabulary
    pipeline.named_steps["vec"].vocabulary_ = vocabulary
    pipeline.named_steps["vec"].fixed_vocabulary_ = True

    return size_after
