#
# Created by Lukas Lüftinger on 2/5/19.
#
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GenotypeRecord:
    """
    Genomic features of a sample referenced by `identifier`.
    """
    identifier: str
    features: List[str]

    def __repr__(self):
        return f"{self.identifier} n_features={len(self.features)}"


@dataclass
class PhenotypeRecord:
    """
    Ground truth labels of sample `identifier`,
    indicating presence/absence of trait `trait_name`:
      - 0 if trait is absent
      - 1 if trait is present
    """
    identifier: str
    trait_name: str
    trait_sign: int

    def __repr__(self):
        return f"{self.identifier} trait({self.trait_name})={self.trait_sign}"


@dataclass
class GroupRecord:
    """
    Group label of sample `identifier`.
    Notes
    -----
    Useful for leave-one-group-out cross-validation (LOGO-CV),
    for example, to take taxonomy into account.
    """
    identifier: str
    group_name: Optional[str]
    group_id: Optional[int]

    def __repr__(self):
        return f"{self.identifier} group({self.group_name})={self.group_id}"


@dataclass
class TrainingRecord(GenotypeRecord, PhenotypeRecord, GroupRecord):
    """
    Sample containing Genotype-, Phenotype- and GroupRecords,
    suitable as machine learning input for a single observation.
    """
    def __repr__(self):
        gr_repr = GenotypeRecord.__repr__(self).split(' ')[1]
        pr_repr = PhenotypeRecord.__repr__(self).split(' ')[1]
        if self.group_name is not None and self.group_id is not None:
            gro_repr = GroupRecord.__repr__(self).split(' ')[1]
        else:
            gro_repr = ''
        return f"id={self.identifier} {' '.join([gr_repr, pr_repr, gro_repr])}"
