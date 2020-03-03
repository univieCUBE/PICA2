from pathlib import Path
from functools import partial

import click


click.option = partial(click.option, show_default=True)


@click.group(short_help='Plotting of results')
def plot():
    """
    Plot results.
    """
    pass


@plot.command()
@click.option('--inputs', type=click.Path(exists=True), required=True, nargs=0,
              help='CCCV output file(s) to plot.')
@click.argument('inputs', nargs=-1)
@click.option('--out', type=click.Path(), help='Output file path. If not given, `.show()` result.')
@click.option('--title', type=str, default='', help='Plot title.')
def cccv(inputs, out, title):
    """Plot CCCV result(s)."""
    from phenotrex.io.flat import load_cccv_accuracy_file
    from phenotrex.util.plotting import compleconta_plot

    conditions = [Path(str(x)).stem for x in inputs]
    cccv_results = [load_cccv_accuracy_file(x) for x in inputs]
    compleconta_plot(cccv_results=cccv_results, conditions=conditions, title=title, save_path=out)


@plot.command('shap-force')
@click.argument('fasta_files', type=click.Path(exists=True), nargs=-1)
@click.option('--genotype', type=click.Path(exists=True),
              required=False, help='Input genotype file.')
@click.option('--classifier', required=True, type=click.Path(exists=True),
              help='Path of pickled classifier file.')
@click.option('--out_prefix', required=True, type=str,
              help='The prefix to generated SHAP force plots.')
@click.option('--n_samples', type=int, default=None,
              help='The nsamples parameter of SHAP. '
                   'Only used by models which utilize a `shap.KernelExplainer` (e.g. TrexSVM).')
@click.option('--verb', is_flag=True)
def shap_force(fasta_files, genotype, classifier, out_prefix, n_samples, verb):
    """
    Generate SHAP force plots for each sample (passed either as FASTA files or as genotype file).
    All plots will be saved at the path `{out_prefix}_{sample_identifier}_force_plot.png`.
    All non-existent directories in the output prefix path will be created as needed.
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    try:
        from phenotrex.transforms import fastas_to_grs
    except ModuleNotFoundError:
        from phenotrex.util.helpers import fail_missing_dependency as fastas_to_grs
    from phenotrex.io.flat import load_genotype_file
    from phenotrex.io.serialization import load_classifier
    from phenotrex.ml.shap_handler import ShapHandler
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
    fs, sv, bv = model.get_shap(gr, nsamples=n_samples)
    sh.add_feature_data(sample_names=[x.identifier for x in gr],
                        features=fs, shaps=sv, base_value=bv)
    for record in tqdm(gr, unit='samples', desc='Generating force plots'):
        sh.plot_shap_force(record.identifier)
        out_path = Path(f'{out_prefix}_{record.identifier}_force_plot.png')
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path)
        plt.close(plt.gcf())
