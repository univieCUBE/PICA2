import click

from phenotrex.cli.compute_genotype import compute_genotype
from phenotrex.cli.train import train
from phenotrex.cli.cv import cv
from phenotrex.cli.cccv import cccv
from phenotrex.cli.predict import predict
from phenotrex.cli.get_weights import get_weights
from phenotrex.cli.plot import plot


class ArtsyMainCli(click.Group):
    def get_help(self, ctx):
        click.secho("""
        ██████╗ ██╗  ██╗███████╗███╗   ██╗ ██████╗ ████████╗██████╗ ███████╗██╗  ██╗
        ██╔══██╗██║  ██║██╔════╝████╗  ██║██╔═══██╗╚══██╔══╝██╔══██╗██╔════╝╚██╗██╔╝
        ██████╔╝███████║█████╗  ██╔██╗ ██║██║   ██║   ██║   ██████╔╝█████╗   ╚███╔╝
        ██╔═══╝ ██╔══██║██╔══╝  ██║╚██╗██║██║   ██║   ██║   ██╔══██╗██╔══╝   ██╔██╗
        ██║     ██║  ██║███████╗██║ ╚████║╚██████╔╝   ██║   ██║  ██║███████╗██╔╝ ██╗
        ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
        """, fg='green')
        return super().get_help(ctx)


@click.group(cls=ArtsyMainCli, context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    pass


cli.add_command(compute_genotype)
cli.add_command(train)
cli.add_command(cv)
cli.add_command(cccv)
cli.add_command(predict)
cli.add_command(get_weights)
cli.add_command(plot)


def main():
    cli()


if __name__ == '__main__':
    main()
