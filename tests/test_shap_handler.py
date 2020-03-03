from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from tests.targets import (first_genotype_accession, first_phenotype_accession)
from phenotrex.io.flat import load_training_files
from phenotrex.io.serialization import load_classifier
from phenotrex.ml.shap_handler import ShapHandler
from . import DATA_PATH

trait_names = [
    "Sulfate_reducer",
    # "Aerobe",
    # "sporulation",
]

classifier_ids = [
    'SVM',
    'XGB',
]


class TestShapHandler:
    @pytest.mark.parametrize("trait_name", trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_get_shap(self, trait_name, classifier_type):
        """
        Get ShapHandler and SHAP data from classifier and genotype file.

        :param trait_name:
        :param classifier_type:
        :return:
        """
        full_path_genotype = DATA_PATH / f"{trait_name}.genotype"
        full_path_phenotype = DATA_PATH / f"{trait_name}.phenotype"
        training_records, genotype, phenotype, group = load_training_files(
            genotype_file=full_path_genotype,
            phenotype_file=full_path_phenotype,
            verb=True)
        assert genotype[0].identifier == first_genotype_accession[trait_name]
        assert phenotype[0].identifier == first_phenotype_accession[trait_name]
        tr = training_records[:3]
        model_path = DATA_PATH / f'{trait_name}_{classifier_type.lower()}.pkl'
        clf = load_classifier(model_path, verb=True)
        sh = ShapHandler.from_clf(clf)
        fs, sv, bv = clf.get_shap(tr, nsamples=50)
        return tr, sh, fs, sv, bv

    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_plot_shap_force(self, trait_name, classifier_type):
        with TemporaryDirectory() as tmpdir:
            tr, sh, fs, sv, bv = self.test_get_shap(trait_name, classifier_type)
            sh.add_feature_data(sample_names=[x.identifier for x in tr],
                                features=fs, shaps=sv, base_value=bv)
            for record in tr:
                sh.plot_shap_force(record.identifier)
                out_path = Path(tmpdir) / '_'.join(
                    ['testing_force_plots', f'{record.identifier}_force_plot.png'])
                out_path.parent.mkdir(exist_ok=True)
                plt.savefig(out_path)
                plt.close(plt.gcf())
            assert len(list(Path(tmpdir).glob('*.png'))) == 3

    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_plot_shap_summary(self, trait_name, classifier_type):
        with TemporaryDirectory() as tmpdir:
            tr, sh, fs, sv, bv = self.test_get_shap(trait_name, classifier_type)
            sh.add_feature_data(sample_names=[x.identifier for x in tr],
                                features=fs, shaps=sv, base_value=bv)
            sh.plot_shap_summary()
            out = Path(tmpdir)/'summary.png'
            plt.savefig(out)
            assert out.is_file()

    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_get_shap_summary(self, trait_name, classifier_type):
        tr, sh, fs, sv, bv = self.test_get_shap(trait_name, classifier_type)
        sh.add_feature_data(sample_names=[x.identifier for x in tr],
                            features=fs, shaps=sv, base_value=bv)
        ss = sh.get_shap_summary()
        print(ss)
        assert len(ss)

    @pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
    @pytest.mark.parametrize('classifier_type', classifier_ids, ids=classifier_ids)
    def test_get_shap_force(self, trait_name, classifier_type):
        tr, sh, fs, sv, bv = self.test_get_shap(trait_name, classifier_type)
        sh.add_feature_data(sample_names=[x.identifier for x in tr],
                            features=fs, shaps=sv, base_value=bv)
        for record in tr:
            sf = sh.get_shap_force(sample_name=record.identifier)
            print(sf)
            assert len(sf)
