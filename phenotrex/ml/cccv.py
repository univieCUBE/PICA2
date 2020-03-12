#
# Created by Lukas Lüftinger on 14/02/2019.
#
import os
import copy
from time import time
from typing import List, Dict, Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from phenotrex.structure.records import TrainingRecord
from phenotrex.transforms.resampling import TrainingRecordResampler
from phenotrex.util.logging import get_logger
from phenotrex.util.helpers import get_x_y_tn
from phenotrex.ml.feature_select import recursive_feature_elimination


class CompleContaCV:
    """
    A class containing all custom completeness/contamination cross-validation functionality.

    :param pipeline: target pipeline which describes the vectorization and estimator/classifier used
    :param scoring_function: Sklearn-like scoring function of crossvalidation. Default: Balanced Accuracy.
    :param cv: Number of folds in crossvalidation. Default: 5
    :param comple_steps: number of steps between 0 and 1 (relative completeness) to be simulated
    :param conta_steps: number of steps between 0 and 1 (relative contamination level) to be simulated
    :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
    :param n_replicates: Number of times the crossvalidation is repeated
    :param reduce_features: toggles feature reduction using recursive feature elimination
    :param n_features: minimal number of features to retain (if feature reduction is used)
    :param random_state: An integer random seed or instance of np.random.RandomState
    """
    def __init__(self, pipeline: Pipeline, scoring_function: Callable = balanced_accuracy_score, cv: int = 5,
                 comple_steps: int = 20, conta_steps: int = 20,
                 n_jobs: int = -1, n_replicates: int = 10, random_state: np.random.RandomState = None,
                 verb: bool = False, reduce_features: bool = False, n_features: int = 10000):
        self.pipeline = pipeline
        self.cv = cv
        self.scoring_method = scoring_function
        self.logger = get_logger(__name__, verb=verb)
        if comple_steps < 1:
            self.logger.warn(f"Completeness steps parameter is out of range: {comple_steps}, was set to 1 instead")
            comple_steps = 1
        if conta_steps < 1:
            self.logger.warn(f"Contamination steps parameter is out of range: {conta_steps}, was set to 1 instead")
            conta_steps = 1

        self.comple_steps = comple_steps
        self.conta_steps = conta_steps
        if n_jobs is not None:
            self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        else:
            self.n_jobs = None
        self.n_replicates = n_replicates
        self.random_state = random_state if type(random_state) is np.random.RandomState \
            else np.random.RandomState(random_state)
        self.reduce_features = reduce_features
        self.n_features = n_features

    def _validate_subset(self, records: List[TrainingRecord], estimator: Pipeline):
        """
        Use a fitted Pipeline to predict scores on resampled test data.
        part of the compleconta crossvalidation where only validation is performed.

        :param records: test records as a List of TrainingRecord objects
        :param estimator: classifier previously trained as a sklearn.Pipeline object
        :return: score
        """
        X, y, tn = get_x_y_tn(records)
        preds = estimator.predict(X)
        score = self.scoring_method(y, preds)
        return score

    def _replicates(self, records: List[TrainingRecord], cv: int = 5,
                    comple_steps: int = 20, conta_steps: int = 20,
                    n_replicates: int = 10):
        """
        Generator function to yield test/training sets which will be fed into subprocesses for _completeness_cv

        :param records: the complete set of TrainingRecords
        :param cv: number of folds in the crossvalidation to be performed
        :param comple_steps: number of steps between 0 and 1 (relative completeness) to be simulated
        :param conta_steps: number of steps between 0 and 1 (relative contamination level) to be simulated
        :param n_replicates: number of replicates for the entire crossvalidation
        :return: parameter list to submit to worker process
        """
        for r in range(n_replicates):
            X, y, tn = get_x_y_tn(records)
            skf = StratifiedKFold(n_splits=cv, random_state=self.random_state)
            fold = 0
            for train_index, test_index in skf.split(X, y):
                fold += 1
                # separate in training set lists:
                training_records = [records[i] for i in train_index]
                test_records = [records[i] for i in test_index]
                starting_message = f"Starting comple/conta replicate {r + 1}/{n_replicates}: fold {fold}"
                yield [test_records, training_records, comple_steps, conta_steps, self.logger.level, starting_message]

    def _completeness_cv(self, param, **kwargs) -> Dict[float, Dict[float, float]]:
        """
        Perform completeness/contamination simulation and testing for one fold.
        This is a separate function only called by run_cccv which spawns
        subprocesses using a ProcessPoolExecutor from concurrent.futures

        :param param: List [test_records, X_train, y_train, comple_steps, conta_steps, starting_message]
            workaround to get multiple parameters into this function. (using processor.map) #TODO find nicer solution?
        """
        # unpack parameters
        test_records, training_records, comple_steps, conta_steps, verb, starting_message = param

        # needed to create a new logger, self.logger not accessible from a different process
        logger = get_logger(__name__, verb=verb)
        logger.info(starting_message)

        classifier = copy.deepcopy(self.pipeline)
        if self.reduce_features:
            recursive_feature_elimination(training_records, classifier, n_features=self.n_features,
                                          random_state=self.random_state)

        X_train, y_train, tn = get_x_y_tn(training_records)
        classifier.fit(X=X_train, y=y_train, **kwargs)

        # initialize the resampler with the test_records only,
        # so the samples are unknown to the classifier
        resampler = TrainingRecordResampler(random_state=self.random_state, verb=False)
        resampler.fit(records=test_records)
        cv_scores = {}
        comple_increment = 1 / comple_steps
        conta_increment = 1 / conta_steps
        for comple in np.arange(0, 1.05, comple_increment):
            comple = np.round(comple, 2)
            cv_scores[comple] = {}
            for conta in np.arange(0, 1.05, conta_increment):
                conta = np.round(conta, 2)
                resampled_set = [resampler.get_resampled(x, comple, conta) for x in test_records]
                cv_scores[comple][conta] = self._validate_subset(resampled_set, classifier)
        return cv_scores

    def run(self, records: List[TrainingRecord]):
        """ Perform completeness/contamination cross-validation.

        :param records: List[TrainingRecords] to perform compleconta-crossvalidation on.
        :return: A dictionary with mean balanced accuracies for each combination: dict[comple][conta]=mba
        """
        # TODO: run compress_vocabulary before?

        self.logger.info(
            "Begin completeness/contamination matrix crossvalidation on training data.")
        t1 = time()
        if self.n_jobs is None:
            cv_scores = map(self._completeness_cv, self._replicates(records, self.cv,
                                                                    self.comple_steps,
                                                                    self.conta_steps,
                                                                    self.n_replicates))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                cv_scores = executor.map(self._completeness_cv,
                                         self._replicates(records, self.cv,
                                                          self.comple_steps,
                                                          self.conta_steps,
                                                          self.n_replicates))
        t2 = time()
        mba = {}
        cv_scores_list = [x for x in cv_scores]

        for comple in cv_scores_list[0].keys():
            mba[comple] = {}
            for conta in cv_scores_list[0][comple].keys():
                single_result = [cv_scores_list[r][comple][conta] for r in
                                 range(self.n_replicates * self.cv)]
                mean_over_fold_and_replicates = np.mean(single_result)
                std_over_fold_and_replicates = np.std(single_result)
                mba[comple][conta] = {"score_mean": mean_over_fold_and_replicates,
                                      "score_sd": std_over_fold_and_replicates}
        self.logger.info(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return mba
