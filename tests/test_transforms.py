import pytest

from Bio.SeqRecord import SeqRecord

from phenotrex.transforms.resampling import TrainingRecordResampler
from phenotrex.transforms.annotation import call_proteins, load_fasta_file
from phenotrex.io.flat import load_training_files
from phenotrex.structure.records import GenotypeRecord

from . import fastas_to_grs, FROM_FASTA, DATA_PATH


predict_files = [
    DATA_PATH/'GCA_000692775_1_trunc2.fna.gz',
    DATA_PATH/'GCA_000692775_1_trunc2.fna',
    DATA_PATH/'GCA_000692775_1_trunc2.faa.gz',
    DATA_PATH/'GCA_000692775_1_trunc2.faa'
]


trait_names = [
    "Sulfate_reducer",
    # "Aerobe",
    # "sporulation",
]


@pytest.mark.parametrize('trait_name', trait_names, ids=trait_names)
def test_resampling(trait_name):
    td, *_ = load_training_files(DATA_PATH/f'{trait_name}.genotype',
                                 DATA_PATH/f'{trait_name}.phenotype')
    trr = TrainingRecordResampler(random_state=2, verb=True)
    trr.fit(td)
    trr.get_resampled(td[0], comple=.5, conta=.5)


@pytest.mark.skipif(not FROM_FASTA, reason='Missing optional dependencies')
def test_call_proteins():
    seqtype, seqs = load_fasta_file(predict_files[1])
    proteins = call_proteins(seqs)
    assert isinstance(proteins[0], SeqRecord)
    print(proteins[0])


@pytest.mark.skipif(not FROM_FASTA, reason='Missing optional dependencies')
@pytest.mark.parametrize('infile', predict_files, ids=['fna-gz', 'fna', 'faa-gz', 'faa'])
def test_compute_genotype(infile):
    gr = fastas_to_grs([infile, ], verb=False)
    assert all(isinstance(x, GenotypeRecord) for x in gr)
    assert all(len(x.features) > 0 for x in gr)
