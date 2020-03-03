from time import time
from typing import List, Tuple, Dict, Callable, Union
from abc import ABC
from abc import abstractmethod
from pprint import pformat
import gc

from scipy.sparse import csr_matrix
import numpy as np

from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.feature_selection import RFECV

from phenotrex.ml.vectorizer import CustomVectorizer
from phenotrex.structure.records import TrainingRecord, GenotypeRecord
from phenotrex.ml.cccv import CompleContaCV
from phenotrex.util.helpers import get_x_y_tn, get_groups
from phenotrex.ml.feature_select import recursive_feature_elimination, compress_vocabulary, \
    DEFAULT_STEP_SIZE, \
    DEFAULT_SCORING_FUNCTION


class TrexClassifier(ABC):
    """
    Abstract base class of Trex classifier.
    """
    scoring_function_mapping = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'f1': f1_score
    }

    @classmethod
    def get_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __init__(self, random_state: int = None, verb: bool = False):
        self.trait_name = None
        self.cccv_result = None
        self.pipeline = None
        self.cv_pipeline = None
        self.logger = None
        self.random_state = np.random.RandomState(random_state)
        self.random_state_init = random_state
        self.verb = verb
        self.vectorizer = CustomVectorizer(binary=True, dtype=np.bool)
        self.default_search_params = None
        self.n_jobs = 1

    def _get_raw_features(self, records: List[GenotypeRecord]) -> csr_matrix:
        """
        Apply the trained vectorizer in the TrexClassifier and return a numpy array suitable for
        inputting into a classifier or SHAP explainer.
        """
        if self.trait_name is None:
            raise RuntimeError('TrexClassifier not fitted.')
        vec = self.pipeline.named_steps['vec']
        X = vec.transform([" ".join(x.features) for x in records])
        return X

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

        extra_explainer_arg = kwargs.pop('train_explainer', None)
        if extra_explainer_arg is not None:
            self.logger.warning(f'{self.__class__.__name__} provides SHAP explanations without '
                                f'training an Explainer. Argument '
                                f'"train_explainer"={extra_explainer_arg} ignored.')
        self.pipeline.fit(X=X, y=y, **kwargs)
        self.logger.info("Classifier training completed.")
        return self

    def predict(self, X: List[GenotypeRecord]) -> Tuple[List[str], np.ndarray]:
        """
        Predict trait sign and probability of each class for each supplied GenotypeRecord.

        :param X: A List of GenotypeRecord for each of which to predict the trait sign
        :return: a Tuple of predictions and probabilities of each class for each GenotypeRecord in X.
        """
        features: List[str] = [" ".join(x.features) for x in X]
        preds = self.pipeline.predict(X=features)
        probas = self.pipeline.predict_proba(
            X=features)  # class probabilities via Platt scaling
        return preds, probas

    @abstractmethod
    def get_feature_weights(self) -> Dict:
        """
        Extract the weights for features from pipeline.

        :return: sorted Dict of feature name: weight
        """
        pass

    def get_shap(self, records: List[GenotypeRecord],
                 nsamples=None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute SHAP (SHapley Additive exPlanations) values for the given input data with the fitted
        TrexClassifier.

        :param records: A list of TrainingRecords or GenotypeRecords.
        :param nsamples: the nsamples parameter to be passed to the Explainer. Only used if the model
                         in question relies on a KernelExplainer (e.g. TrexSVM).
        :returns: transformed feature array, computed shap values and expected value.
        """
        pass

    def parameter_search(self, records: List[TrainingRecord],
                         search_params: Dict[str, List] = None,
                         cv: int = 5, scoring: str = DEFAULT_SCORING_FUNCTION,
                         n_jobs: int = -1, n_iter: int = 10,
                         return_optimized: bool = False):
        """
        Perform stratified, randomized parameter search. If desired, return a new class instance
        with optimized training parameters.

        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param search_params: A dictionary of iterables of possible model training parameters.
                              If None, use default search parameters for the given classifier.
        :param scoring: Scoring function of crossvalidation. Default: Balanced Accuracy.
        :param cv: Number of folds in crossvalidation. Default: 5
        :param n_jobs: Number of parallel jobs. Default: -1 (All processors used)
        :param n_iter: Number of grid points to evaluate. Default: 10
        :param return_optimized: Whether to return a ready-made classifier
                                 with the optimized params instead of a dictionary of params.
        :return: A dictionary containing best found parameters or an optimized class instance.
        """
        if n_jobs != 1 and self.n_jobs > 1:
            self.logger.info(f'Will use selected classifier parallelism instead of multithreading.')
            n_jobs = self.n_jobs

        t1 = time()
        self.logger.info(f'Performing randomized parameter search.')
        X, y, tn = get_x_y_tn(records)
        if search_params is None:
            search_params = self.default_search_params

        vec = clone(self.cv_pipeline.named_steps['vec'])
        clf = clone(self.cv_pipeline.named_steps['clf'])

        X_trans = vec.fit_transform(X)
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        rcv = RandomizedSearchCV(estimator=clf,
                                 scoring=scoring,
                                 param_distributions=search_params,
                                 n_jobs=n_jobs,
                                 n_iter=n_iter,
                                 cv=cv, iid=False,
                                 verbose=1 if self.verb else 0)

        rcv.fit(X_trans, y=y)
        best_params = rcv.best_params_
        t2 = time()
        gc.collect()  # essential due to imperfect memory management of XGBoost sklearn interface

        self.logger.info(f'Optimized params:\n{pformat(best_params)}')
        self.logger.info(f'{np.round(t2 - t1)} sec elapsed during parameter search.')
        if return_optimized:
            self.logger.info(f'Returning optimized instance of {self.__class__.__name__}.')
            return self.get_instance(**best_params,
                                     random_state=self.random_state_init,
                                     verb=self.verb)
        return best_params

    def crossvalidate(self, records: List[TrainingRecord], cv: int = 5,
                      scoring: Union[str, Callable] = DEFAULT_SCORING_FUNCTION,
                      n_jobs=-1,
                      n_replicates: int = 10, groups: bool = False,
                      # TODO: add more complex scoring/reporting, e.g. AUC
                      reduce_features: bool = False,
                      n_features: int = 10000,
                      demote=False, **kwargs) -> Tuple[float, float, np.ndarray]:
        """
        Perform cv-fold crossvalidation or leave-one(-group)-out validation if groups == True

        :param records: List[TrainingRecords] to perform crossvalidation on.
        :param scoring: String identifying scoring function of crossvalidation, or Callable.
                        If a callable is passed, it must take two parameters `y_true` and `y_pred`
                        (iterables of true and predicted class labels, respectively) and return a
                        (numeric) score.
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
        if n_jobs != 1 and self.n_jobs > 1:
            self.logger.info(f'Will use selected classifier parallelism instead of multithreading.')
            n_jobs = self.n_jobs

        scoring_func = scoring if hasattr(scoring, '__call__') else self.scoring_function_mapping.get(scoring)
        assert scoring_func is not None, f'invalid or missing scoring function: {scoring}.'
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
            splitting_strategy = StratifiedKFold(n_splits=cv, shuffle=True,
                                                 random_state=self.random_state)
            group_ids = None

        for i in range(n_replicates):
            inner_cv = StratifiedKFold(n_splits=cv, shuffle=True,
                                       random_state=self.random_state)
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
                score = scoring_func(y[ts], y_pred)
                scores.append(score)
            log_function(f"Finished replicate {i + 1} of {n_replicates}")

        misclassifications /= n_replicates
        score_mean, score_sd = float(np.mean(scores)), float(np.std(scores))
        t2 = time()
        log_function(f"Cross-validation completed.")
        log_function(f"Total duration of cross-validation: {np.round(t2 - t1, 2)} seconds.")
        return score_mean, score_sd, misclassifications

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
        if n_jobs != 1 and self.n_jobs > 1:
            self.logger.info(f'Will use internal classifier parallelism instead of subprocessing.')
            n_jobs = self.n_jobs

        cccv = CompleContaCV(pipeline=self.cv_pipeline, cv=cv,
                             comple_steps=comple_steps, conta_steps=conta_steps,
                             n_jobs=n_jobs, n_replicates=n_replicates,
                             random_state=self.random_state, verb=self.verb,
                             reduce_features=reduce_features, n_features=n_features)
        score_dict = cccv.run(records=records)
        self.cccv_result = score_dict
        return score_dict
