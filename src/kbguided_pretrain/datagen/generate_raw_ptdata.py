import os
from tqdm import tqdm
import re
from random import shuffle
import numpy
import copy
import sys
import pandas as pd
import json
import joblib
import random
from collections import defaultdict
import ipdb

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
        # for _ in range(100000):
            yield line
            line = f.readline()
    return

def conv(x):
    if isinstance(x, list) or isinstance(x, set):
        return [conv(xx) for xx in x]
    x = x.strip().lower()
    for ch in ',.;{}[]()+-_*/?!`\"\'=%></':
        x = x.replace(ch, ' ')
    return ' '.join([a for a in x.split() if a])


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
    
    def generate_name_list_set(self, semantic_type, source_onto):
        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        semantic_reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        rel_reader = byLineReader(os.path.join(self.umls_path, "MRREL." + self.type))

        '''
        self.cui2triple is a dictionary which projects CUI of head entity to relation (label) and tail entity CUIs.
        '''
        self.cui_in_onto = set()
        self.cui2triple = defaultdict(set)
        self.relStat = defaultdict(lambda:0)
        self.cui2pref = dict()

        for line in tqdm(rel_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            if len(l[7])>0:
                cui = l[0]
                rel = l[7]
                self.relStat[rel] += 1
                object = l[4]
                self.cui2triple[cui].add((rel, object))
        # number of triples in total in the pre-training datax
        print("the length of cui in the umls dump is", len(self.cui2triple)) 
        totalRel = sum(self.relStat.values())
        for rel in self.relStat.keys():
            self.relStat[rel] = self.relStat[rel]/totalRel


        '''
        self.cui_in_onto is a set which includes only CUI which are in the target ontology
        self.cui2pref: a dictionary, cui ---> list() of sysnonyms 
         '''
        self.cui2pref = dict()
        self.cui_in_onto = set()
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
                    # which 
                    self.cui_in_onto.update([cui])
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

        for cui in copy.deepcopy(list(self.cui2pref.keys())):
            if cui not in self.cuis_in_semtc or cui not in self.cui_in_onto:
                self.cui2pref.pop(cui)
        
        syn_count = 0
        for cui in self.cui2pref:
            self.cui2pref[cui] = list(set(conv(self.cui2pref[cui])))
            syn_count += len(self.cui2pref[cui])
        
        print("cui count:", len(self.cui2pref))
        print("synonyms count:", syn_count)
    
    def generate_syn_des(self):
        name_reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        def_reader = byLineReader(os.path.join(self.umls_path, "MRDEF." + self.type))
        self.cui2description = dict()
        cuiset = set()
        auiset = set()
        for line in tqdm(def_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            if cui in self.cui2pref:
                cuiset.update([l[0]])
        for line in tqdm(name_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            aui = l[7]
            lang = l[1]
            if lang != 'ENG' and cui in cuiset:
                auiset.update([aui])

        def_reader = byLineReader(os.path.join(self.umls_path, "MRDEF." + self.type))
        for line in tqdm(def_reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            aui = l[1]
            defi = l[5].lower()
            if cui in cuiset and aui not in auiset:
                if cui not in self.cui2description:
                    self.cui2description[cui] = [defi]
                else:
                    self.cui2description[cui].append(defi)
        
        des_count = 0
        for cui in self.cui2description:
            des_count += len(self.cui2description[cui])

        print('number of description:', des_count)

def generate_pair(y, mentions, select_scheme):
    if select_scheme == 'random':
        return random.choice(mentions)
    elif select_scheme == 'sample':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        # print(similarity_estimate.shape)
        return np.random.choice(mentions, 1, p = similarity_estimate/np.sum(similarity_estimate))[0]
    elif select_scheme == 'most_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmax()]
    elif select_scheme == 'least_sim':
        similarity_estimate = cal_similarity_tfidf(mentions, y, vectorizer)
        return mentions[similarity_estimate.argmin()]
    else:
        print('Wrong mention selection scheme input!!!')

template_sets = ['is defined as', 'is described as', 'are the definations of', 'describe', 'define']
template_sets_nodef = ['are the synonyms of', 'indicate the same concept as', 'has synonyms, such as', 'refers to the same concept as']
template_sets_nosyn = ['is', 'is the same as', 'is', 'is the same as']
def create_line(prefix, mention, context, special_tokens, template_choice):
    if prefix:
        des = ' '.join([special_tokens[0], mention, special_tokens[1], template_choice, context])
    else:
        des = ' '.join([context, template_choice, special_tokens[0], mention, special_tokens[1]])
    return des


# create the synthetic text based on cui and triple it connects to
rels = ["has_entry_version", "has_sort_version", "entry_version_of", "permuted_term_of", "sort_version_of", "has_permuted_term", "mapped_to"]
def trip2txt(mention, triples, cui2syns, mode, relStat):
    '''
    triples: triple this entity is connected to
    '''
    n_sample = 5
    synText = []
    if len(triples)<=100:
        for pair in triples:
            if pair[1] in cui2syns and pair[0] not in rels:
                text = []
#                text.extend(pair[0].split('_'))
                text.extend([pair[0]])
                text.append(random.choice(cui2syns[pair[1]]) + '.')
                synText.append(text)
        if len(synText) == 0:
            return synText

        if mode == 'Tri':
            for idx in range(len(synText)):
                synText[idx][0] = synText[idx][0].split('_')
            synText = " ".join(synText)

        if mode == 'ByLine':
            if len(synText) <= n_sample:
                for idx in range(len(synText)):
                    rel = synText[idx][0].split('_')
                    tail = synText[idx][-1]
                    text = []
                    text.extend(rel)
                    text.append(tail)
                    synText[idx] = text
                synText = [ " ".join(i) for i in synText]
                return synText

            pairDir = defaultdict(list)
            # create a dictionary for key ---> [tails]
            for pair in synText:
                rel, tail = pair[0], pair[1]
                pairDir[rel].append(rel +'__:__'+ tail)
            seq, probs = list(), list()
            for rel in pairDir.keys():
                tails = pairDir[rel]
                seq.append(tails)
                probs.append(relStat[rel] * n_sample) 
            # sums up to 1
            probs = [i/sum(probs) for i in probs]
            seq = numpy.array(seq)
            indices = numpy.arange(seq.shape[0])

            output = seq[numpy.random.choice(a=indices, p=probs)]
            output = [item for item in output]

            while len(output) <  n_sample:
                addi = seq[numpy.random.choice(a=indices, p=probs)]
                addi = [item for item in output]
                output.extend(addi)

            output = output[:n_sample-1]

            out= list()
            for pair in output:
                rel, tail = pair.split('__:__')
                text = []
                text.extend(rel.split('_'))
                text.append(tail)
                out.append( " ".join(text))
            return out

    else:
        for pair in triples:
            if random.randint(0, len(triples)) <= 99:
                if pair[1] in cui2syns and pair[0] not in rels:
                    text = []
                    text.extend([pair[0]])
                    text.append(random.choice(cui2syns[pair[1]]) + '.')
                    synText.append(text)
        if len(synText) == 0:
            return synText

        if mode == 'Tri':
            synText = " ".join(synText)

        if mode == 'ByLine':
            if len(synText) <= n_sample:
                for idx in range(len(synText)):
                    rel = synText[idx][0].split('_')
                    tail = synText[idx][-1]
                    text = []
                    text.extend(rel)
                    text.append(tail)
                    synText[idx] = text
                synText = [ " ".join(i) for i in synText]
                return synText

            pairDir = defaultdict(list)
            # create a dictionary for key ---> [tails]
            for pair in synText:
                rel, tail = pair[0], pair[1]
                pairDir[rel].append(rel +'__:__'+ tail)
            seq, probs = list(), list()
            for rel in pairDir.keys():
                tails = pairDir[rel]
                seq.append(tails)
                probs.append(relStat[rel] * n_sample) 
            # sums up to 1
            probs = [i/sum(probs) for i in probs]
            seq = numpy.array(seq)
            indices = numpy.arange(seq.shape[0])

            output = seq[numpy.random.choice(a=indices, p=probs)]
            output = [item for item in output]

            while len(output) <  n_sample:
                addi = seq[numpy.random.choice(a=indices, p=probs)]
                addi = [item for item in output]
                output.extend(addi) 
            output = output[:n_sample-1]

            out= list()
            for pair in output:
                rel, tail = pair.split('__:__')
                text = []
                text.extend(rel.split('_'))
                text.append(tail)
                out.append( " ".join(text))
            return out

    return synText

def prepare_final_pretraindata(cui2defs, cui2syns, cui2triple, relStat, special_tokens = None, select_scheme = 'random', mode = 'ByLine'):
    '''
    mode = str (ByLine, Syn, Tri) inficating different mode of pre-training
    '''
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    output = []
    for cui in tqdm(cui2syns):
        triples = cui2triple[cui]
        for syn in cui2syns[cui]:
            # generate description
            #synthesize a description line if cui does not have a description saved in KB.
            if cui not in cui2defs:
                if len(cui2syns[cui]) > 1:
                    pending_set = copy.deepcopy(cui2syns[cui])
                    pending_set.remove(syn)
                    mention = generate_pair(syn, pending_set, select_scheme)
                    random.shuffle(pending_set)
                    idx = random.randint(0, 3)
                    des = create_line(idx>1, mention, ', '.join(pending_set[:3]), special_tokens, template_sets_nodef[idx])
                else:
                    mention = syn
                    idx = random.randint(0, 3)
                    des = create_line(idx>1, mention, syn, special_tokens, template_sets_nosyn[idx])
            else:
                idx = random.randint(0, 4)
                if len(cui2syns[cui]) > 1:
                    pending_set = copy.deepcopy(cui2syns[cui])
                    pending_set.remove(syn)
                    mention = generate_pair(syn, pending_set, select_scheme)
                else:
                    mention = syn
                random.shuffle(cui2defs[cui])
                idx = random.randint(0, 3)
                des = create_line(idx<2, mention, ' '.join(cui2defs[cui][:2]), special_tokens, template_sets[idx])
                tks = tokenizer(des)['input_ids']
                if len(tks) > 512:
                    if idx < 2:
                        des = tokenizer.decode(tks[:512])
                    else:
                        des = tokenizer.decode(tks[-512:])

            output.append([cui, mention, syn, des, None])
            if mode == 'Syn':
                continue
        # syntext is the linearilized triples
            if len(triples) >= 1:
                synText = trip2txt(mention, triples, cui2syns, mode, relStat)
                # in case there's no text generated
                if len(synText) >= 1:
                    if mode == 'Tri':
                        tks = tokenizer(synText)['input_ids']
                        if len(tks) > 512:
                            synText = tokenizer.decode(tks[:512])
                            # more than <s></s>
                        output.append([cui, mention, syn, des, synText])
                    if mode == 'ByLine':
                        for line in synText:
                            tks = tokenizer(line)['input_ids']
                            if len(tks) > 512:
                                line = tokenizer.decode(tks[:512])
                                # more than <s></s>
                            output.append([cui, mention, syn, des, line])


    random.shuffle(output)
    return output
                

if __name__ ==  '__main__':



    semantic_type = set(['T005','T007','T017','T022','T031','T033','T037','T038','T058','T062','T074',
                    'T082','T091','T092','T097','T098','T103','T168','T170','T201','T204'])
    import os
    print(os.getcwd())
    semantic_type_ontology = pd.read_csv('STY.csv') # TUI->STRING mapping table
    semantic_type_size = 0
    while len(semantic_type)!=semantic_type_size:
        semantic_type_size = len(semantic_type)
        for i in range(len(semantic_type_ontology)):
            if semantic_type_ontology['Parents'][i][-4:] in semantic_type:
                semantic_type.update([semantic_type_ontology['Class ID'][i][-4:]])
    source_onto = ['CPT','FMA','GO','HGNC','HPO','ICD10','ICD10CM','ICD9CM','MDR','MSH','MTH',
                    'NCBI','NCI','NDDF','NDFRT','OMIM','RXNORM','SNOMEDCT_US', "SNOMEDCT_CT"]
    UMLS = UMLS('/export/home/yan/el/', only_load_dict = True)

    UMLS.generate_name_list_set(semantic_type, source_onto)
    UMLS.generate_syn_des()

    print('cuicount', len(UMLS.cui2pref))
    print('defcount', len(UMLS.cui2description))
    count = 0
    for cui in UMLS.cui2pref:
        if len(UMLS.cui2pref[cui]) >=2:
            count += 1
    print(count)

    output = prepare_final_pretraindata(UMLS.cui2description, UMLS.cui2pref, UMLS.cui2triple, UMLS.relStat, special_tokens = ["START", "END"], select_scheme = 'random', mode = 'ByLine')
    shuffle(output)
    f = None
    if not os.path.exists('./test0_raw_data/'):
        os.makedirs('./test0_raw_data/')
    for i in tqdm(range(len(output))):
        if i%100000 == 0:
            if f:
                f.close()
            f = open('./test0_raw_data/data_'+str(i//100000).rjust(3,'0')+'.txt', 'w')
        f.write(json.dumps(output[i])+'\n')