from functools import partial

import click

from phenotrex.cli.generic_opt import universal_options, common_train_options, param_options
from phenotrex.cli.clf_opt import xgb_options, svm_options

click.option = partial(click.option, show_default=True)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]),
             short_help="Classifier training and serialization")
def train():
    """
    Perform training and saving of clf.
    Requires a .genotype and a .phenotype file.
    """
    pass


@train.command()
@universal_options
@common_train_options
@param_options
@xgb_options
def xgb(*args, **kwargs):
    """Train XGB model."""
    from phenotrex.cli.generic_func import generic_train
    generic_train('xgb', *args, **kwargs)


@train.command()
@universal_options
@common_train_options
@param_options
@svm_options
def svm(*args, **kwargs):
    """Train SVM model."""
    from phenotrex.cli.generic_func import generic_train
    generic_train('svm', *args, **kwargs)
