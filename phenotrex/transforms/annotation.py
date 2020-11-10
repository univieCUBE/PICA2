import os
import sys
from collections import namedtuple
from typing import List
from pathlib import Path
from pkg_resources import resource_filename
from tempfile import NamedTemporaryFile
from subprocess import check_call, DEVNULL
from concurrent.futures import ProcessPoolExecutor

import torch
from deepnog.learning.inference import predict
from deepnog.utils import create_df, get_weights_path
from deepnog.utils import load_nn, set_device
from deepnog.data.dataset import ProteinDataset, ProteinIterator
from Bio.SeqIO import SeqRecord, parse, write
from tqdm.auto import tqdm

from phenotrex.io.flat import load_fasta_file
from phenotrex.structure.records import GenotypeRecord


PRODIGAL_BIN_SUFFIX = {'win32': 'windows.exe', 'darwin': 'osx.10.9.5'}.get(sys.platform, 'linux')
PRODIGAL_BIN_PATH = resource_filename('phenotrex', f'bin/prodigal.{PRODIGAL_BIN_SUFFIX}')
DEEPNOG_ARCH = 'deepencoding'
DEEPNOG_VALID_CONFIG = {
    ('eggNOG5', 1),
    ('eggNOG5', 2),
}


class PreloadedProteinIterator(ProteinIterator):
    """Hack ProteinDataset to load from list directly."""
    def __init__(
        self, protein_list: List[SeqRecord], aa_vocab, format
    ):
        self.iterator = (x for x in protein_list)
        self.vocab = aa_vocab
        self.format = format
        self.start = 0
        self.pos = None
        self.step = 0
        self.sequence = namedtuple('sequence', ['index', 'id', 'string', 'encoded'])


class PreloadedProteinDataset(ProteinDataset):
    """Hack ProteinDataset to load from list directly."""
    def __init__(self, protein_list: List[SeqRecord]):
        super().__init__(sequences=protein_list)
        self.protein_list = protein_list

    def __iter__(self):
        return PreloadedProteinIterator(
            protein_list=self.protein_list, aa_vocab=self.vocab, format=self.f_format
        )


def fastas_to_grs(
    fasta_files: List[str], verb: bool = False,
    n_threads: int = None
) -> List[GenotypeRecord]:
    """
    Perform GenotypeRecord calculation for a list of FASTA files. Apply process-based parallelism
    since gene annotation scales well with cores.

    :param fasta_files: a list of DNA and/or protein FASTA files to be converted into
                        GenotypeRecords.
    :param verb: Whether to display progress of annotation with tqdm.
    :param n_threads: Number of parallel threads. Default, use all available CPU cores.
    :returns: A list of GenotypeRecords corresponding with supplied FASTA files.
    """
    n_threads = min(os.cpu_count(), n_threads) if n_threads is not None else os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        if len(fasta_files) > 1 and verb:
            annotations = tqdm(
                executor.map(fasta_to_gr, fasta_files),
                total=len(fasta_files),
                desc='Annotating with DeepNOG',
                unit='file'
            )
        else:
            annotations = executor.map(fasta_to_gr, fasta_files)
    return list(annotations)


def fasta_to_gr(fasta_file: str, verb: bool = False) -> GenotypeRecord:
    """
    Given a fasta file, determine whether gene calling is required (DNA fasta) or if deepnog can be
    applied directly (protein fasta). If required, perform prodigal gene call and return
    GenotypeRecord (output of deepnog) of the file.

    :param fasta_file: A DNA or protein fasta file to be converted into GenotypeRecord.
    :param verb: Whether to display progress of annotation with tqdm.
    :returns: A single GenotypeRecord representing the sample.
    """
    fname = Path(str(fasta_file)).name
    seqtype, seqs = load_fasta_file(fasta_file)
    if seqtype == 'protein':
        return annotate_with_deepnog(fname, seqs, verb=verb)
    else:
        return annotate_with_deepnog(fname, call_proteins(seqs), verb=verb)


def call_proteins(seqs: List[SeqRecord]) -> List[SeqRecord]:
    """
    Perform protein calling with prodigal.

    :param seqs: A list of DNA fasta SeqRecords.
    :returns: a list of protein fasta SeqRecords suitable for annotation with deepnog.
    """
    with NamedTemporaryFile(mode='w', delete=False) as fna_file:
        write(seqs, fna_file, format='fasta')
    with NamedTemporaryFile(mode='r', delete=False) as faa_file:
        check_call([
            PRODIGAL_BIN_PATH,
            '-i', fna_file.name,
            '-a', faa_file.name
        ], stderr=DEVNULL, stdout=DEVNULL)
        parsed = list(parse(faa_file, 'fasta'))
    os.unlink(fna_file.name)
    os.unlink(faa_file.name)  # cannot re-use .name of open NamedTemporaryFile under Win32
    return parsed


def annotate_with_deepnog(
    identifier: str,
    protein_list: List[SeqRecord],
    database: str = 'eggNOG5',
    tax_level: int = 2,
    verb: bool = True
) -> GenotypeRecord:
    """
    Perform calling of EggNOG5 clusters on a list of SeqRecords belonging to a sample, using deepnog.

    :param identifier: The name associated with the sample.
    :param protein_list: A list of SeqRecords containing protein sequences.
    :param database: Orthologous group/family database to use.
    :param tax_level: The NCBI taxon ID of the taxonomic level to use from the given database.
    :param verb: Whether to print verbose progress messages.
    :returns: a GenotypeRecord suitable for use with phenotrex.
    """
    if not (database, tax_level) in DEEPNOG_VALID_CONFIG:
        raise RuntimeError(f'Unknown database and/or tax level: {database}/{tax_level}')

    device = set_device('auto')
    torch.set_num_threads(1)
    weights_path = get_weights_path(
        database=database, level=str(tax_level), architecture=DEEPNOG_ARCH,
    )
    model_dict = torch.load(weights_path, map_location=device)
    model = load_nn(
        architecture=DEEPNOG_ARCH,
        model_dict=model_dict,
        device=device,
    )
    class_labels = model_dict['classes']
    dataset = PreloadedProteinDataset(protein_list)
    preds, confs, ids, indices = predict(
        model, dataset, device, batch_size=1, num_workers=1, verbose=3 if verb else 0
    )
    threshold = float(model.threshold) if hasattr(model, 'threshold') else None
    df = create_df(class_labels, preds, confs, ids, indices, threshold=threshold, )

    cogs = [x for x in df.prediction.unique() if x]
    feature_type_str = f'{database}-tax-{tax_level}'
    return GenotypeRecord(identifier=identifier, feature_type=feature_type_str, features=cogs)
