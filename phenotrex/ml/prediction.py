import pandas as pd

from phenotrex.io.flat import load_genotype_file, DEFAULT_TRAIT_SIGN_MAPPING
from phenotrex.io.serialization import load_classifier
from phenotrex.ml.shap_handler import ShapHandler

try:
    from phenotrex.transforms import fastas_to_grs
except ModuleNotFoundError:
    from phenotrex.util.helpers import fail_missing_dependency as fastas_to_grs


def predict(fasta_files=tuple(), genotype=None, classifier=None,
            out_explain_per_sample=None, out_explain_summary=None,
            shap_n_samples=None, n_max_explained_features=None, verb=False):
    """
    Predict phenotype from a set of (possibly gzipped) DNA or protein FASTA files
    or a single genotype file. Optionally, compute SHAP explanations individually and/or summarily
    for the predicted samples.

    NB: Genotype computation is highly expensive and performed on the fly on FASTA files.
    For increased speed when predicting multiple phenotypes, create a .genotype file to reuse
    with the command `compute-genotype`.

    NB: As opposed to XGB models where they are trivially available, computing SHAP explanations
    on SVM models entails training a model-agnostic KernelExplainer which is highly costly (dozens
    to hundreds of seconds per sample if using a somewhat reasonable value for `shap_n_samples`).

    :param fasta_files: An iterable of fasta file paths
    :param genotype: A genotype file path
    :param classifier: A pickled classifier file path
    :param out_explain_per_sample: Where to save the most influential features by SHAP for each
                                   predicted sample.
    :param out_explain_summary: Where to save the SHAP summary of the predictions.
    :param shap_n_samples: The nsamples parameter -
                           only used by models which incorporate a `shap.KernelExplainer`.
    :param n_max_explained_features: How many of the most influential features by SHAP to consider.
    :param verb: Whether to show progress of fasta file annotation.
    """
    if not len(fasta_files) and genotype is None:
        raise RuntimeError('Must supply FASTA file(s) and/or single genotype file for prediction.')
    if len(fasta_files):
        grs_from_fasta = fastas_to_grs(fasta_files, n_threads=None, verb=verb)
    else:
        grs_from_fasta = []

    grs_from_file = load_genotype_file(genotype) if genotype is not None else []
    gr = grs_from_fasta + grs_from_file

    model = load_classifier(filename=classifier, verb=verb)
    if out_explain_per_sample is not None or out_explain_summary is not None:
        sh = ShapHandler.from_clf(model)
        fs, sv, bv = model.get_shap(gr, nsamples=shap_n_samples)
        sh.add_feature_data(sample_names=[x.identifier for x in gr],
                            features=fs, shaps=sv, base_value=bv)
        if out_explain_per_sample is not None:
            shap_df = pd.concat([
                sh.get_shap_force(x.identifier, n_max_features=n_max_explained_features) for x in gr
            ], axis=0)
            shap_df.to_csv(out_explain_per_sample, sep='\t', index=False)
        if out_explain_summary is not None:
            sum_df = sh.get_shap_summary(n_max_explained_features)
            sum_df.to_csv(out_explain_summary, sep='\t', index=False)

    preds, probas = model.predict(X=gr)
    translate_output = {trait_id: trait_sign for trait_sign, trait_id in
                        DEFAULT_TRAIT_SIGN_MAPPING.items()}
    print(f"# Trait: {model.trait_name}")
    print("Identifier\tTrait present\tConfidence")
    for record, result, probability in zip(gr, preds, probas):
        print(f"{record.identifier}\t{translate_output[result]}\t{str(round(probability[result], 4))}")
