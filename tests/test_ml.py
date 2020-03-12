import sys
import json

import pytest
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.targets import (first_genotype_accession, first_phenotype_accession, first_groups_accession,
                           cv_scores_trex, num_of_features_compressed, num_of_features_uncompressed)
from phenotrex.io.flat import (load_training_files,
                               write_weights_file, write_params_file, write_misclassifications_file)
from phenotrex.io.serialization import save_classifier
from phenotrex.ml import TrexSVM, TrexXGB
from phenotrex.util.helpers import get_x_y_tn
from phenotrex.ml.feature_select import recursive_feature_elimination, compress_vocabulary
from phenotrex.ml.prediction import predict

from . import DATA_PATH, FROM_FASTA


RANDOM_STATE = 2


trait_names = [
    "Sulfate_reducer",
    # "Aerobe",
    # "sporulation",
]

classifiers = [
    TrexSVM,
    TrexXGB,
]

classifier_ids = [
    'SVM',
    'XGB',
]

cv_folds = [
    5,
]

scoring_methods = [
    "balanced_accuracy",
    "f1",
    # "accuracy",
]

predict_files = [
    (DATA_PATH/'GCA_000692775_1_trunc2.fna.gz', ),
    (DATA_PATH/'GCA_000692775_1_trunc2.faa.gz', ),
    (DATA_PATH/'GCA_000692775_1_trunc2.fna.gz', DATA_PATH/'GCA_000692775_1_trunc2.faa.gz')
]


class TestTrexClassifier:

    @staticmethod
    def _round_nested_dict(d, decimal=1):
        return json.loads(json.dumps(d), parse_float=lambda x: round(float(x), decimal))

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    def test_load_training_files(self, trait_name):
        """
        Test training data loading. Check/catch invalid file formats.
        :param trait_name:
        :return:
        """
        full_path_genotype = DATA_PATH / f"{trait_name}.genotype"
        full_path_phenotype = DATA_PATH / f"{trait_name}.phenotype"
        full_path_groups = DATA_PATH / f"{trait_name}.taxids"
        training_records, genotype, phenotype, group = load_training_files(
            genotype_file=full_path_genotype,
            phenotype_file=full_path_phenotype,
            groups_file=full_path_groups,
            verb=True)
        assert genotype[0].identifier == first_genotype_accession[trait_name]
        assert phenotype[0].identifier == first_phenotype_accession[trait_name]
        assert group[0].identifier == first_groups_accession[trait_name]
        return training_records, genotype, phenotype, group

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    @pytest.mark.parametrize("use_shaps", [True, False], ids=['shap', 'noshap'])
    def test_train(self, trait_name, classifier, use_shaps):
        """
        Test TrexClassifier training. Using different traits.
        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        clf.train(records=training_records, train_explainer=use_shaps)
        with TemporaryDirectory() as tmpdir:
            clf_path = Path(tmpdir)/'classifier.pkl'
            weights_path = Path(tmpdir)/'weights.rank'
            save_classifier(clf,  clf_path)
            weights = clf.get_feature_weights()
            write_weights_file(weights_file=weights_path, weights=weights)
            assert clf_path.is_file()
            assert weights_path.is_file()

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("cv", cv_folds, ids=[str(x) for x in cv_folds])
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    @pytest.mark.parametrize("use_groups", [True, False], ids=['logo', 'nologo'])
    def test_crossvalidate(self, trait_name, cv, classifier, use_groups):
        """
        Test default crossvalidation of TrexClassifier class.
        Using several different traits, cv folds, and scoring methods.
        Compares with dictionary cv_scores.

        :param trait_name:
        :param cv:
        :param scoring:
        :param classifier:
        :param use_groups:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        score_pred = clf.crossvalidate(records=training_records,
                                       cv=cv, scoring=scoring_methods[0],
                                       groups=use_groups)[:2]
        if classifier.identifier in cv_scores_trex and not use_groups:
            score_target = cv_scores_trex[classifier.identifier][trait_name][cv][scoring_methods[0]]
            np.testing.assert_almost_equal(actual=score_pred, desired=score_target, decimal=1)
        with TemporaryDirectory() as tmpdir:
            misclass_path = Path(tmpdir)/'misclassifications.tsv'
            write_misclassifications_file(misclass_path, training_records,
                                          score_pred, use_groups=use_groups)
            assert misclass_path.is_file()

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_parameter_search(self, trait_name, classifier):
        """
        Test randomized parameter search.

        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        clf_opt = clf.parameter_search(records=training_records, n_iter=5, return_optimized=False)
        assert isinstance(clf_opt, dict)
        with TemporaryDirectory() as tmpdir:
            param_path = Path(tmpdir)/'params.json'
            write_params_file(param_path, clf_opt)
            assert param_path.is_file()

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_compleconta_cv(self, trait_name, classifier):
        """
        Perform compleconta-cv for each trait name using TrexClassifier class.
        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        cccv_scores = clf.crossvalidate_cc(records=training_records, cv=5, comple_steps=3,
                                           conta_steps=3)
        assert isinstance(cccv_scores, dict)

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_get_feature_names(self, trait_name, classifier):
        """
        Get feature names of classifier.

        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        clf.train(training_records)
        fweights = clf.get_feature_weights()
        print(fweights)
        print(len(fweights))

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_get_shap_values(self, trait_name, classifier):
        """
        Get shap values associated with the training data.
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        clf.train(training_records)
        # nsamples only used by TrexSVM; reduced number of samples due to TrexSVM
        raw_features, shaps, bias = clf.get_shap(training_records[:5], nsamples=50)
        print(shaps.shape)
        print(bias)

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("classifier", classifiers, ids=classifier_ids)
    def test_compress_vocabulary(self, trait_name, classifier):
        """
        Perform feature compression tests

        :param trait_name:
        :param classifier:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        clf = classifier(verb=True, random_state=RANDOM_STATE)
        compress_vocabulary(records=training_records, pipeline=clf.cv_pipeline)
        vec = clf.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) == num_of_features_uncompressed[trait_name]

        X, y, tn = get_x_y_tn(training_records)
        X_trans = vec.transform(X)

        # check if number of unique features is matching
        assert X_trans.shape[1] == num_of_features_compressed[trait_name]

        # check if all samples still have at least one feature present
        one_is_zero = False
        non_zero = X_trans.nonzero()
        for x in non_zero:
            if len(x) == 0:
                one_is_zero = True
        assert not one_is_zero

    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize("n_features", [10_000])
    def test_recursive_feature_elimination(self, trait_name, n_features):
        """
        Perform feature compression tests only for SVM; counterindicated for XGB.
        :param trait_name:
        :return:
        """
        training_records, genotype, phenotype, group = self.test_load_training_files(trait_name)
        svm = TrexSVM(verb=True, random_state=RANDOM_STATE)
        recursive_feature_elimination(records=training_records,
                                      pipeline=svm.cv_pipeline,
                                      step=0.01,
                                      n_features=n_features,
                                      )
        vec = svm.cv_pipeline.named_steps["vec"]
        vec._validate_vocabulary()

        # check if vocabulary is set properly
        assert vec.fixed_vocabulary_

        # check if length of vocabulary is matching
        assert len(vec.vocabulary_) >= n_features

        X, y, tn = get_x_y_tn(training_records)
        X_trans = vec.transform(X)

        # check if number of unique features is matching
        assert X_trans.shape[1] >= n_features

        # check if all samples still have at least one feature present
        one_is_zero = False
        non_zero = X_trans.nonzero()
        for x in non_zero:
            if len(x) == 0:
                one_is_zero = True
        assert not one_is_zero

    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_predict_from_genotype(self, trait_name, classifier_type):
        model_path = DATA_PATH/f'{trait_name}_{classifier_type.lower()}.pkl'
        genotype_file = DATA_PATH/f'{trait_name}.genotype'
        print(predict(classifier=model_path, genotype=genotype_file))

    @pytest.mark.skipif(not FROM_FASTA, reason='Missing optional dependencies')
    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('fasta_files', predict_files, ids=['fna', 'faa', 'fna+faa'])
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_predict_from_fasta(self, trait_name, classifier_type, fasta_files):
        model_path = DATA_PATH/f'{trait_name}_{classifier_type.lower()}.pkl'
        print(predict(fasta_files=fasta_files, classifier=model_path))
