import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from kbguided_pretrain.datagen.generate_raw_ptdata import UMLS

def get_mappings():
    semantic_type = {'T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 'T038', 'T058', 'T062', 'T074', 'T082',
                     'T091', 'T092', 'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204'}
    semantic_type_ontology = pd.read_csv('../kbguided_pretrain/datagen/STY.csv')  # TUI->STRING mapping table
    semantic_type_size = 0
    while len(semantic_type) != semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10', 'ICD10CM', 'ICD9CM', 'MDR', 'MSH', 'MTH',
                   'NCBI', 'NCI', 'NDDF', 'NDFRT', 'OMIM', 'RXNORM', 'SNOMEDCT_US']

    umls = UMLS("../../META", only_load_dict=True)
    medic_mapping, snomed_mapping, mesh_mapping, all_sources = umls.generate_mappings()

    print(len(snomed_mapping))
    print(len(mesh_mapping))
    print(sorted(all_sources))
    return mesh_mapping

def get_synonym_mapping():
    semantic_type = set(['T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 'T038', 'T058', 'T062', 'T074',
                         'T082', 'T091', 'T092', 'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204'])
    semantic_type_ontology = pd.read_csv('../kbguided_pretrain/datagen/STY.csv')  # TUI->STRING mapping table
    semantic_type_size = 0
    while len(semantic_type) != semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['CPT', 'FMA', 'GO', 'HGNC', 'HPO', 'ICD10', 'ICD10CM', 'ICD9CM', 'MDR', 'MSH', 'MTH',
                   'NCBI', 'NCI', 'NDDF', 'NDFRT', 'OMIM', 'RXNORM', 'SNOMEDCT_US']


    umls = UMLS("../../META", only_load_dict=True)
    umls.generate_name_list_set(semantic_type, source_onto)
    synonym_mapping = defaultdict(set)
    for key, value in umls.cui2pref.items():
        for x in value:
            synonym_mapping[x].add(key)
    return synonym_mapping, umls.cui2triple


def check_ask_a(synonym_mapping, cui2triple):
    ask_patient_path = Path("../finetune/AskAPatient")
    for file in ask_patient_path.glob("*.train.txt"):
        pass

def get_cdr_coocc(mesh_mapping: dict):
    training_dataset = Path("../finetune/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt")
    current_entities = []
    entities_per_example = []
    for line in training_dataset.open():
        split_lines = line.split("\t")
        if line == "\n":
            entities_per_example.append(current_entities)
            current_entities = []
        if len(split_lines) == 1:
            continue
        identifier = split_lines[-1]
        if identifier.startswith("D"):
            try:
                current_entities.append(mesh_mapping[identifier.strip()])
            except:
                pass
    return entities_per_example

def get_ncbi_coocc(mesh_mapping: dict):
    training_dataset = Path("../finetune/ncbi/NCBItrainset_corpus.txt")
    current_entities = []
    entities_per_example = []
    for line in training_dataset.open():
        split_lines = line.split("\t")
        if line == "\n":
            entities_per_example.append(current_entities)
            current_entities = []
        if len(split_lines) == 1:
            continue
        identifier = split_lines[-1]
        if identifier.startswith("D"):
            try:
                current_entities.append(mesh_mapping[identifier.strip()])
            except:
                pass
    return entities_per_example

def main():
    mesh_mapping = get_mappings()
    synonym_mapping, cui2triple = get_synonym_mapping()

    existing_triples = {}
    for key, triples in cui2triple.items():
        for triple in triples:
            existing_triples[(key, triple[1])] = triple[0]
    entities_per_example = get_cdr_coocc(mesh_mapping)

    most_occurring_relations = defaultdict(int)
    cooccurred = 0
    for entities in entities_per_example:
        for entity in entities:
            for entity_ in entities:
                if entity_ != entity:
                    if (entity, entity_) in existing_triples:
                        most_occurring_relations[existing_triples[(entity, entity_)]] += 1
                        cooccurred += 1
    print(cooccurred / len(entities_per_example))
    print(json.dumps({key: value/len(entities_per_example) for key, value in most_occurring_relations.items()}, indent=4))

    entities_per_example = get_ncbi_coocc(mesh_mapping)

    most_occurring_relations = defaultdict(int)
    cooccurred = 0
    for entities in entities_per_example:
        for entity in entities:
            for entity_ in entities:
                if entity_ != entity:
                    if (entity, entity_) in existing_triples:
                        most_occurring_relations[existing_triples[(entity, entity_)]] += 1
                        cooccurred += 1
    print(cooccurred / len(entities_per_example))
    print(json.dumps({key: value / len(entities_per_example) for key, value in most_occurring_relations.items()},
                     indent=4))


if __name__ == '__main__':
    main()
