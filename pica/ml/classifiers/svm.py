#!/usr/bin/env python3
#
# Created by Lukas Lüftinger on 2/5/19.
#
from time import time
from typing import List, Tuple, Dict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV

from pica.ml.vectorizer import CustomVectorizer
from pica.structure.records import TrainingRecord, GenotypeRecord
from pica.ml.cccv import CompleContaCV
from pica.ml.trex_classifier import TrexClassifier
from pica.util.logging import get_logger
from pica.util.helpers import get_x_y_tn, get_groups
from pica.ml.feature_select import recursive_feature_elimination, compress_vocabulary, DEFAULT_STEP_SIZE,\
    DEFAULT_SCORING_FUNCTION


class TrexSVM(TrexClassifier):
    """
    Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
    sklearn.svm.LinearSVC.
    Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.

    :param random_state: A integer randomness seed for a Mersienne Twister (see np.random.RandomState)
    :param kwargs: Any additional named arguments are passed to the XGBClassifier constructor.
    """

    identifier = 'SVM'

    def __init__(self, C: float = 5., penalty: str = "l2", tol: float = 1.,
                 random_state: int = None, verb=False,
                 *args, **kwargs):
        super().__init__(random_state=random_state, verb=verb)

        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.logger = get_logger(__name__, verb=verb)

        if self.penalty == "l1":
            self.dual = False
        else:
            self.dual = True

        classifier = LinearSVC(C=self.C, tol=self.tol, penalty=self.penalty, dual=self.dual,
                               class_weight="balanced", random_state=self.random_state, **kwargs)

        self.pipeline = Pipeline(steps=[
            ("vec", self.vectorizer),
            ("clf", CalibratedClassifierCV(classifier, method="sigmoid", cv=5))
        ])
        self.cv_pipeline = Pipeline(steps=[
            ("vec", self.vectorizer),
            ("clf", classifier)
        ])

    def _get_coef_(self, pipeline: Pipeline = None) -> np.array:
        r"""
        Interface function to get `coef\_` from classifier used in the pipeline specified
        this might be useful if we switch the classifier, most of them already have a `coef\_` attribute


        :param pipeline: pipeline from which the classifier should be used
        :return: `coef\_` for feature weight report
        """
        if not pipeline:
            pipeline = self.pipeline

        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return_weights = clf.coef_
        else:  # assume calibrated classifier
            weights = np.array([c.base_estimator.coef_[0] for c in clf.calibrated_classifiers_])
            return_weights = np.median(weights, axis=0)
        return return_weights

    def get_feature_weights(self) -> Dict:
        """
        Extract the weights for features from pipeline.

        :return: sorted Dict of feature name: weight
        """
        # TODO: find different way to feature weights that is closer to the real weight used for classification
        # get weights directly from the CalibratedClassifierCV object.
        # Each classifier has numpy array .coef_ of which we simply take the mean
        # this is not necessary the actual weight used in the final classifier, but enough to determine importance
        if self.trait_name is None:
            self.logger.error("Pipeline is not fitted. Cannot retrieve weights.")
            return {}

        mean_weights = self._get_coef_()

        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()

        # decompress
        weights = {feature: mean_weights[i] for feature, i in names}

        # sort by absolute value
        sorted_weights = {feature: weights[feature] for feature in sorted(weights, key=lambda key: abs(weights[key]),
                                                                          reverse=True)}
        # TODO: weights should be adjusted if multiple original features were grouped together. probably not needed
        #  if we rely on feature selection in near future
        return sorted_weights

class PICASVM:
    """
    Class which encapsulates a sklearn Pipeline of CountVectorizer (for vectorization of features) and
    LinearSVC wrapped in CalibratedClassifierCV for provision of probabilities via Platt scaling.
    Provides train() and crossvalidate() functionality equivalent to train.py and crossvalidateMT.py.

    :param C: Penalty parameter C of the error term. See LinearSVC documentation.
    :param penalty: Specifies the norm used in the penalization. See LinearSVC documentation.
    :param tol: Tolerance for stopping criteria. See LinearSVC documentation.
    :param random_state: A integer randomness seed for a Mersienne Twister (see np.random.RandomState)
    :param kwargs: Any additional named arguments are passed to the LinearSVC constructor.
    """
    def __init__(self,
                 C: float = 5.,
                 penalty: str = "l2",
                 tol: float = 1.,
                 random_state: int = None,
                 verb=False,
                 *args, **kwargs):
        self.trait_name = None
        self.cccv_result = None  # result of compleconta-crossvalidation saved in object so it gets pickled

        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.logger = get_logger(__name__, verb=verb)
        self.verb = verb

        if self.penalty == "l1":
            self.dual = False
        else:
            self.dual = True

        vectorizer = CustomVectorizer(binary=True, dtype=np.bool)
        classifier = LinearSVC(C=self.C, tol=self.tol, penalty=self.penalty, random_state=self.random_state,
                               dual=self.dual, class_weight="balanced", **kwargs)

        self.pipeline = Pipeline(steps=[
            ("vec", vectorizer),
            ("clf", CalibratedClassifierCV(classifier, method="sigmoid", cv=5))
        ])
        self.cv_pipeline = Pipeline(steps=[
            ("vec", vectorizer),
            ("clf", classifier)
        ])

    def train(self, records: List[TrainingRecord], reduce_features: bool = False,
              n_features: int = 10000, **kwargs):
        """
        Fit CountVectorizer and train LinearSVC on a list of TrainingRecord.

        :param records: a List[TrainingRecord] for fitting of CountVectorizer and training of LinearSVC.
        :param reduce_features: toggles feature reduction using recursive feature elimination
        :param n_features: minimum number of features to retain when reducing features
        :param kwargs: additional named arguments are passed to the fit() method of Pipeline.
        :returns: Whether the Pipeline has been fitted on the records.
        """

        self.logger.info("Begin training classifier.")
        X, y, tn = get_x_y_tn(records)
        if self.trait_name is not None:
            self.logger.warning("Pipeline is already fitted. Refusing to fit again.")
            return False

        if reduce_features:
            self.logger.info("using recursive feature elimination as feature selection strategy")
            # use non-calibrated classifier
            recursive_feature_elimination(records, self.cv_pipeline, n_features=n_features)
            compress_vocabulary(records, self.pipeline)

        self.trait_name = tn

        self.pipeline.fit(X=X, y=y, **kwargs)
        self.logger.info("Classifier training completed.")
        return True

    def crossvalidate(self, records: List[TrainingRecord], cv: int = 5,
                      scoring: str = DEFAULT_SCORING_FUNCTION, n_jobs=-1,
                      n_replicates: int = 10, groups: bool = False,
                      # TODO: add more complex scoring/reporting, e.g. AUC
                      reduce_features: bool = False, n_features: int = 10000,
                      demote=False, **kwargs) -> Tuple[float, float, np.ndarray]:
        """
        Perform cv-fold crossvalidation or leave-one(-group)-out validation if groups == True

        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
        :param n_replicates: Number of replicates of the crossvalidation
        :param groups: If True, use group information stored in records for splitting. Otherwise,
            stratify split according to labels in records. This also resets n_replicates to 1.
        :param reduce_features: toggles feature reduction using recursive feature elimination
        :param n_features: minimum number of features to retain when reducing features
        :param demote: toggles logger that is used. if true, msg is written to debug else info
        :param kwargs: Unused
        :return: A list of mean score, score SD, and the percentage of misclassifications per sample
        """

        log_function = self.logger.debug if demote else self.logger.info
        t1 = time()
        X, y, tn = get_x_y_tn(records)

        # unfortunately RFECV does not work with pipelines (need to use the vectorizer separately)
        self.cv_pipeline.fit(X, y)
        vec = self.cv_pipeline.named_steps["vec"]
        clf = self.cv_pipeline.named_steps["clf"]

        if not vec.vocabulary:
            vec.fit(X)
        X_trans = vec.transform(X)

        misclassifications = np.zeros(len(y))
        scores = []

        if groups:
            log_function("Begin Leave-One-Group-Out validation on training data.")
            splitting_strategy = LeaveOneGroupOut()
            group_ids = get_groups(records)
            n_replicates = 1
        else:
            log_function("Begin cross-validation on training data.")
            splitting_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            group_ids = None

        for i in range(n_replicates):
            inner_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            outer_cv = splitting_strategy
            for tr, ts in outer_cv.split(X_trans, y, groups=group_ids):
                if reduce_features:
                    est = RFECV(estimator=clf, cv=inner_cv, n_jobs=n_jobs,
                                step=DEFAULT_STEP_SIZE, min_features_to_select=n_features,
                                scoring=DEFAULT_SCORING_FUNCTION)
                else:
                    est = clf
                est.fit(X_trans[tr], y[tr])
                y_pred = est.predict(X_trans[ts])
                mismatch = np.logical_xor(y[ts], y_pred)
                mismatch_indices = ts[np.where(mismatch)]
                misclassifications[mismatch_indices] += 1
                score = balanced_accuracy_score(y[ts], y_pred)
                scores.append(score)
            log_function(f"Finished replicate {i + 1} of {n_replicates}")

        misclassifications /= n_replicates
        score_mean, score_sd = float(np.mean(scores)), float(np.std(scores))
        t2 = time()
        log_function(f"Cross-validation completed.")
        log_function(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return score_mean, score_sd, misclassifications

    def predict(self, X: List[GenotypeRecord]) -> Tuple[List[str], np.ndarray]:
        """
        Predict trait sign and probability of each class for each supplied GenotypeRecord.

        :param X: A List of GenotypeRecord for each of which to predict the trait sign
        :return: a Tuple of predictions and probabilities of each class for each GenotypeRecord in X.
        """
        features: List[str] = [" ".join(x.features) for x in X]
        preds = self.pipeline.predict(X=features)
        probas = self.pipeline.predict_proba(X=features)  # class probabilities via Platt scaling
        return preds, probas

    def get_coef_(self, pipeline: Pipeline = None) -> np.array:
        r"""
        Interface function to get `coef\_` from classifier used in the pipeline specified
        this might be useful if we switch the classifier, most of them already have a `coef\_` attribute


        :param pipeline: pipeline from which the classifier should be used
        :return: `coef\_` for feature weight report
        """
        if not pipeline:
            pipeline = self.pipeline

        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return_weights = clf.coef_
        else:  # assume calibrated classifier
            weights = np.array([c.base_estimator.coef_[0] for c in clf.calibrated_classifiers_])
            return_weights = np.median(weights, axis=0)
        return return_weights

    def get_feature_weights(self) -> Dict:
        """
        Extract the weights for features from pipeline.

        :return: tuple of lists: feature names and weights
        """
        # TODO: find different way to feature weights that is closer to the real weight used for classification
        # get weights directly from the CalibratedClassifierCV object.
        # Each classifier has numpy array .coef_ of which we simply take the mean
        # this is not necessary the actual weight used in the final classifier, but enough to determine importance
        if self.trait_name is None:
            self.logger.error("Pipeline is not fitted. Cannot retrieve weights.")
            return {}

        mean_weights = self.get_coef_()

        # get original names of the features from vectorization step, they might be compressed
        names = self.pipeline.named_steps["vec"].get_feature_names()

        # decompress
        weights = {feature: mean_weights[i] for feature, i in names}

        # sort by absolute value
        sorted_weights = {feature: weights[feature] for feature in sorted(weights, key=lambda key: abs(weights[key]),
                                                                          reverse=True)}
        # TODO: weights should be adjusted if multiple original features were grouped together. probably not needed
        #  if we rely on feature selection in near future

        return sorted_weights

    def crossvalidate_cc(self, records: List[TrainingRecord], cv: int = 5,
                         comple_steps: int = 20, conta_steps: int = 20,
                         n_jobs: int = -1, n_replicates: int = 10,
                         reduce_features: bool = False, n_features: int = 10000):
        """
        Instantiates a CompleContaCV object, and calls its run_cccv method with records. Returns its result.

        :param records: List[TrainingRecord] on which completeness_contamination_CV is to be performed
        :param cv: number of folds in StratifiedKFold split
        :param comple_steps: number of equidistant completeness levels
        :param conta_steps: number of equidistant contamination levels
        :param n_jobs: number of parallel jobs (-1 for n_cpus)
        :param n_replicates: Number of times the crossvalidation is repeated
        :param reduce_features: toggles feature reduction using recursive feature elimination
        :param n_features: selects the minimum number of features to retain (if feature reduction is used)
        :return: A dictionary with mean balanced accuracies for each combination: dict[comple][conta]=mba
        """

        cccv = CompleContaCV(pipeline=self.cv_pipeline, cv=cv,
                             comple_steps=comple_steps, conta_steps=conta_steps,
                             n_jobs=n_jobs, n_replicates=n_replicates,
                             random_state=self.random_state, verb=self.verb,
                             reduce_features=reduce_features, n_features=n_features)
        score_dict = cccv.run(records=records)
        self.cccv_result = score_dict
        return score_dict