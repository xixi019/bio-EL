import os
from tqdm import tqdm
import re
from random import shuffle
import pickle
import copy
import sys
import pandas as pd
import json
import joblib
import random
import ipdb

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return

class UMLS(object):

    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=False, debug=False):
        self.debug = debug
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()
        # self.load()
        if not only_load_dict:
            self.load_rel()
            self.load_sty()

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"
    
    # based on semantic type and source ontology you want to keep, generate 
    def generate_name_list_set(self, semantic_type, source_onto):

        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        semantic_reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        rel_reader = byLineReader(os.path.join(self.umls_path, "MRREL." + self.type))

        '''
        self.cui2triple is a dictionary which projects CUI of head entity to relation (label and )tail entity CUIs.
        '''
        self.cui_in_onto = set()
        self.cui2triple, self.cui2pref = dict(), dict()

        for line in tqdm(rel_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            if len(l[7])>0:
                cui = l[0]
                rel = l[7]
                object = l[4]
                if cui not in self.cui2triple:
                    self.cui2triple[cui] = []
                else:
                    if (rel, object) not in self.cui2triple[cui]:
                        self.cui2triple[cui].append((rel, object))
        # number of triples in total in the pre-training datax
        print("the length of cui in the umls dump ", len(self.cui2triple)) 

        '''
        self.cui_in_onto is a set which includes only CUI which are in the target ontology
        self.cui2pref: a dictionary, cui ---> list() of strings of the label
        '''
        for line in tqdm(name_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            source = l[11]
            string = l[14]
            ispref = l[6]
            if lang == "ENG":
                if cui in self.cui2pref:
                    self.cui2pref[cui].append(string)
                else:
                    self.cui2pref[cui] = [string]
                if source in source_onto:
                    self.cui_in_onto.update([cui])

        '''
        self.cuis_in_semtc: a dict()  cui -->  label, if the type of that entity is in "semantic_type".
        '''
        self.cuis_in_semtc = {}
        for line in tqdm(semantic_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            semantic = l[1]
            type_str = l[3].lower()
            if semantic in semantic_type:
                self.cuis_in_semtc[cui] = type_str

        for cui in copy.deepcopy(list(self.cui2triple.keys())):
            if cui not in self.cuis_in_semtc or cui not in self.cui_in_onto:
                self.cui2triple.pop(cui)
        print(f"{len(self.cui2triple)} entitys exit in pre-training data.")
        
        rel_count = 0
        
        for cui in self.cui2triple:
            rel_count += len(self.cui2triple[cui])

        print("cui count:", len(self.cui2triple))
        print("triples count:", rel_count)

# create the synthetic text based on cui and triple it connects to
rels = ["has_entry_version", "mapped_to", "has_sort_version", "entry_version_of", "permuted_term_of", "sort_version_of"]
def create_line(cui, triples, cui2syns, special_tokens):
    synText = [special_tokens[0]]
    if len(triples)<=100:
        for pair in triples:
            if cui in cui2syns and pair[1] in cui2syns and pair[0] not in rels:
                synText.append(cui2syns[cui].capitalize())
                synText.extend(pair[0].split('_'))
                synText.append(cui2syns[pair[1]]+'.')
            else:
                pass

        synText.append(special_tokens[1])                
        synText = " ".join(synText)
    else:
        for pair in triples:
            if random.randint(0, len(triples)) <= 99:
                if cui in cui2syns and pair[1] in cui2syns and pair[0] not in rels:
                    synText.append(cui2syns[cui].capitalize())
                    synText.extend(pair[0].split('_'))
                    synText.append(cui2syns[pair[1]]+".")
                else:
                    pass

        synText.append(special_tokens[1])                
        synText = " ".join(synText)
    return synText

# transform triples into text form
def prepare_final_pretraindata(cui2syns, cui2triples, special_tokens = None, select_scheme = 'random'):
    '''
    cui2defs: cui2 definition
    '''
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    output = []
    for cui in tqdm(cui2triples.keys()):
        mention = cui2syns[cui]
        triples = cui2triples[cui]
        synText = create_line(cui, triples, cui2syns, special_tokens)
        tks = tokenizer(synText)['input_ids']
        if len(tks) > 600:
            synText = tokenizer.decode(tks[:650])
           # else:
           #     synText = tokenizer.decode(tks[-700:])
        if len(synText) >= len('START END'):
            output.append([cui, mention, mention, synText])
    random.shuffle(output)
    return output


if __name__ ==  '__main__':

    # add all semantic types 
    semantic_type = set(['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074',
                    'T082','T091','T092','T097','T098','T103','T168','T170','T201','T204'])
    semantic_type_ontology = pd.read_csv('STY.csv') # TUI->STRING mapping table
    semantic_type_size = 0
    while len(semantic_type)!=semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    
    source_onto = ['CPT','FMA','GO','HGNC','HPO','ICD10','ICD10CM','ICD9CM','MDR','MSH','MTH',
                    'NCBI','NCI','NDDF','NDFRT','OMIM','RXNORM','SNOMEDCT_US']
    UMLS = UMLS('/export/home/yan/infhome/el/', only_load_dict = True)

    UMLS.generate_name_list_set(semantic_type, source_onto)

    print('cuicount', len(UMLS.cui2triple))
    # generate the corpora, which is a list of sequences.
    output = prepare_final_pretraindata(UMLS.cuis_in_semtc, UMLS.cui2triple, special_tokens = ["START", "END"])
    shuffle(output)
    f = None
    ipdb.set_trace()
    if not os.path.exists('./raw_data/'):
        os.makedirs('./raw_data/')
    for i in tqdm(range(len(output))):
        if i%100000 == 0:
            if f:
                f.close()
            f = open('./raw_data/data_'+str(i//100000).rjust(3,'0')+'.txt', 'w')
        f.write(json.dumps(output[i])+'\n')








