from functools import partial

import click

from phenotrex.cli.generic_opt import universal_options, common_cv_options, param_options
from phenotrex.cli.clf_opt import xgb_options, svm_options

click.option = partial(click.option, show_default=True)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]),
             short_help="Crossvalidation over completeness/contamination range")
def cccv():
    """
    Perform nested, stratified crossvalidation over a range of simulated completeness
    and contamination values for the training data.
    Optionally, save returned scores for each grid point.
    """
    pass


def cccv_options(f):
    """CCCV-specific CLI options."""
    f = click.option('--out', type=click.Path(), required=True,
                     help='Output file path for CCCV results.')(f)
    f = click.option('--conta-steps', type=int, default=20,
                     help='Number of equidistant contamination levels to resample to.')(f)
    f = click.option('--comple-steps', type=int, default=20,
                     help='Number of equidistant completeness levels to resample to.')(f)
    return f


@cccv.command('xgb')
@universal_options
@common_cv_options
@param_options
@cccv_options
@xgb_options
def xgb(*args, **kwargs):
    """Perform Completeness/Contamination CV on XGB model."""
    from phenotrex.cli.generic_func import generic_cccv
    generic_cccv('xgb', *args, **kwargs)


@cccv.command('svm')
@universal_options
@common_cv_options
@param_options
@cccv_options
@svm_options
def svm(*args, **kwargs):
    """Perform Completeness/Contamination CV on SVM model."""
    from phenotrex.cli.generic_func import generic_cccv
    generic_cccv('svm', *args, **kwargs)
