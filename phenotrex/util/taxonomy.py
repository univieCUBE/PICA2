#
# Created by Patrick Hyden on 04/03/2019
#

from ete3 import NCBITaxa
from typing import Dict, List, Tuple

DEFAULT_AUTO_SELECTED_RANK = "family"


def get_taxonomic_group_mapping(group_ids: List[str], selected_rank: str) -> Tuple[Dict, Dict]:
    """
    Function to create a mapping from NCBI-taxon ids to groups which are used to split the provided
    training records into training and validation sets

    :param group_ids: List of identifiers that should be NCBI taxon ids
    :param selected_rank: selected standard rank determining on which level the set is split in
                          training and validation-set
    :return: Mapping of input taxon_ids as string and groups as integers
    """
    ncbi = NCBITaxa()
    standard_ranks = ["superkingdom", "phylum", "class", "order", "family", "genus", "species"]
    if not selected_rank.lower() in standard_ranks:
        selected_rank = auto_select_rank(group_ids)

    taxon_ids_set = set(group_ids)
    taxon_ancestor_mapping = {}

    for taxon in taxon_ids_set:
        lineage = ncbi.get_lineage(int(taxon))
        ids_of_ranks = ncbi.get_rank(lineage)
        taxon_ancestor_mapping[taxon] = 0   # fall-back value if sample does not have an entry on this level
        for ancestor_id, rank in ids_of_ranks.items():
            if rank == selected_rank:
                taxon_ancestor_mapping[taxon] = ancestor_id

    ancestor_ids = set(taxon_ancestor_mapping.values())
    ancestor_names = ncbi.get_taxid_translator(ancestor_ids)
    ancestor_names[0] = "unknown"
    ancestor_enumeration = {ancestor_id: x for x, ancestor_id in enumerate(ancestor_ids)}

    group_name_mapping = {taxon: ancestor_names[taxon_ancestor_mapping[taxon]] for taxon in group_ids}
    group_id_mapping = {taxon: ancestor_enumeration[taxon_ancestor_mapping[taxon]] for taxon in group_ids}

    return group_name_mapping, group_id_mapping


def auto_select_rank(group_ids: List[str]) -> str:
    """
    Placeholder function to select a taxonomic rank splitting based on provided data

    :param group_ids:
    :return: selected rank
    """
    # TODO: implement some auto detection method, if needed
    return DEFAULT_AUTO_SELECTED_RANK
