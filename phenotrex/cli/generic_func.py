from pprint import pformat

from phenotrex.io.flat import (load_training_files, load_genotype_file, load_params_file,
                               write_weights_file, write_params_file,
                               write_misclassifications_file,
                               write_cccv_accuracy_file)
from phenotrex.io.serialization import save_classifier, load_classifier
from phenotrex.util.logging import get_logger
from phenotrex.ml import TrexSVM, TrexXGB, ShapHandler
from phenotrex.transforms.annotation import fastas_to_grs

CLF_MAPPER = {'svm': TrexSVM, 'xgb': TrexXGB}
logger = get_logger("phenotrex", verb=True)


def _fix_uppercase(kwargs):
    """
    Properly handle uppercase arguments which are normalized by click.
    """
    if 'c' in kwargs:
        kwargs['C'] = kwargs.pop('c')
    return kwargs


def generic_train(type, genotype, phenotype, verb, weights, out,
                  n_features=None, params_file=None, *args, **kwargs):
    """
    Train and save a TrexClassifier model.
    """
    kwargs = _fix_uppercase(kwargs)
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               verb=verb)
    if params_file is not None:
        loaded_params = load_params_file(params_file)
        logger.info(f'Parameters loaded from file:')
        logger.info('\n' + pformat(loaded_params))
        kwargs = {**kwargs, **loaded_params}
    clf = CLF_MAPPER[type](verb=verb, *args, **kwargs)

    reduce_features = True if n_features is not None else False
    clf.train(records=training_records, reduce_features=reduce_features, n_features=n_features)
    if weights:
        weights = clf.get_feature_weights()
        weights_file_name = f"{out}.rank"
        write_weights_file(weights_file=weights_file_name, weights=weights)
    save_classifier(obj=clf, filename=out, verb=verb)


def generic_cv(type, genotype, phenotype, folds, replicates, threads, verb, optimize=False,
               optimize_out=None, optimize_n_iter=None, groups=None, rank=None, out=None,
               n_features=None, params_file=None,
               *args, **kwargs):
    """
    Estimate model performance by cross-validation.
    Optionally, perform parameter search and save found parameters.
    """
    kwargs = _fix_uppercase(kwargs)
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               groups_file=groups,
                                               selected_rank=rank,
                                               verb=verb)
    if params_file is not None:
        loaded_params = load_params_file(params_file)
        logger.info(f'Parameters loaded from file:')
        logger.info('\n' + pformat(loaded_params))
        kwargs = {**kwargs, **loaded_params}  # TODO: should loaded params have precendence?

    clf = CLF_MAPPER[type](verb=verb, *args, **kwargs)

    if optimize:
        assert optimize_out is not None, 'No savepath for found parameters passed.'
        logger.info(f'Optimizing parameters...')
        found_params = clf.parameter_search(training_records, n_iter=optimize_n_iter)
        params = {**kwargs, **found_params}
        write_params_file(optimize_out, params)
        logger.info(f'Optimized parameters written to file {optimize_out}.')
        clf = CLF_MAPPER[type](verb=verb, *args, **params)

    reduce_features = True if n_features is not None else False
    use_groups = groups is not None
    logger.info(f'Running CV...')
    score_mean, score_sd, misclass = clf.crossvalidate(records=training_records, cv=folds,
                                                       n_replicates=replicates, groups=use_groups,
                                                       n_jobs=threads,
                                                       reduce_features=reduce_features,
                                                       n_features=n_features,
                                                       demote=not verb)
    logger.info(f"CV score: {round(score_mean, 4)} +/- {round(score_sd, 4)}")
    if out is not None:
        write_misclassifications_file(out, training_records, misclass, use_groups=use_groups)


def generic_cccv(type, genotype, phenotype, folds, replicates, threads, comple_steps, conta_steps,
                 verb, groups=None, rank=None, optimize=False, out=None, n_features=None,
                 params_file=None,
                 *args, **kwargs):
    """
    Perform crossvalidation over a range of simulated completeness/contamination values,
    and save output.
    """
    kwargs = _fix_uppercase(kwargs)
    assert groups is None, 'Usage of LOGO in CCCV not currently implemented.'
    assert not optimize, 'Parameter search over CCCV not currently implemented.'
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               verb=verb)
    if params_file is not None:
        loaded_params = load_params_file(params_file)
        logger.info(f'Parameters loaded from file:')
        logger.info('\n' + pformat(loaded_params))
        kwargs = {**kwargs, **loaded_params}  # TODO: should loaded params have precendence?
    clf = CLF_MAPPER[type](verb=verb, *args, **kwargs)
    reduce_features = True if n_features is not None else False
    cccv = clf.crossvalidate_cc(records=training_records, cv=folds, n_replicates=replicates,
                                comple_steps=comple_steps, conta_steps=conta_steps,
                                n_jobs=threads, reduce_features=reduce_features,
                                n_features=n_features)
    write_cccv_accuracy_file(out, cccv)


def generic_compute_shaps(fasta_files, genotype, classifier, n_samples, verb):
    """
    Given a genotype file and/or a collection of possibly gzipped FASTA files as well as a
    phenotrex classifier, collect genotype information from both, get SHAP information about the
    genotypes using the classifier, and return a finished ShapHandler object as well as the list
    of GenotypeRecords created.
    """
    if not len(fasta_files) and genotype is None:
        raise RuntimeError(
            'Must either supply FASTA file(s) or single genotype file for prediction.')
    if len(fasta_files):
        grs_from_fasta = fastas_to_grs(fasta_files, n_threads=None, verb=verb)
    else:
        grs_from_fasta = []

    grs_from_file = load_genotype_file(genotype) if genotype is not None else []
    gr = grs_from_fasta + grs_from_file

    model = load_classifier(filename=classifier, verb=verb)
    sh = ShapHandler.from_clf(model)
    try:
        fs, sv, bv = model.get_shap(gr, nsamples=n_samples)
    except TypeError:
        raise RuntimeError('This TrexClassifier is not capable of generating SHAP explanations.')
    sh.add_feature_data(sample_names=[x.identifier for x in gr],
                        features=fs, shaps=sv, base_value=bv)
    return sh, gr
