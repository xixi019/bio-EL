from tqdm import tqdm
from typing import Tuple
import pickle
import random
import numpy as np
import os
import copy
from multiprocessing import Process
from transformers import BartTokenizer
import json


# not working
def truncate_input_sequence(document, max_num_tokens):
    try:
        total_length = len(sum(document, []))
    except:
        total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else:
        tokens_to_trunc = total_length - max_num_tokens
        if tokens_to_trunc > 0:
            document[-1] = document[-1][:-tokens_to_trunc]
        
#            if not len(sum(document, [])) <= 512:
#                print(document)
#                print(len(document[0]), len(document[1]))
#                print(total_length, tokens_to_trunc)
        return document

class TokenInstance:
    """ This TokenInstance is a obect to have the basic units of data that should be
        extracted from the raw text file and can be consumed by any BERT like model.
    """
    def __init__(self, tokens_x, tokens_y, prefixlen, lang="en"):
        self.tokens_x = tokens_x
        self.tokens_y = tokens_y
        self.prefixlen = prefixlen
        self.lang = lang

    def get_values(self):
        return (self.tokens_x, self.tokens_y, self.prefixlen)

    def get_lang(self):
        return self.lang

class PretrainingDataCreator:
    def __init__(self,
                 path,
                 tokenizer,
                 max_seq_length,
                 readin = 2000000,
                 dupe_factor = 5,
                 small_seq_prob = 0.1,
                 ):
        
        raise NotImplementedError('not implemented!!!')

    def __len__(self):
        return self.len

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def create_training_instance(self, index):
        
        raise NotImplementedError('not implemented')


class BioBARTPretrainDataCreator(PretrainingDataCreator):
    def __init__(self,
                 path,
                 tokenizer,
                 max_seq_length: int = 512,
                 readin: int = 2000000,
                 dupe_factor: int = 5,
                 sentence_permute_prob: float = 0.8,
                 small_seq_prob: float = 0.1):

        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob
        self.sentence_permute_prob = sentence_permute_prob

        documents = []
        instances = []
        new_doc = False
        with open(path, 'r', encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = json.loads(line.strip('\n'))
                # x is description
                x = tokenizer.tokenize(line[3])
                # y is synonym information to the decoder
                if line[4]:
                    y = [tokenizer.tokenize(' '+line[1]), tokenizer.tokenize(line[4])]
                else:
                    y = [tokenizer.tokenize(' '+line[1]+' is'), tokenizer.tokenize(' '+line[2])]

                # if the length of synonym is 300, the model would be asked to predict  nothing since we limit the input size to 300 due to OOM issues.
                if len(y[0]) >= 300:
                    continue
                cui = line[0]
#                if len(sum(y, [])) >= 512:
                documents.append([x, y, cui])
        documents = [x for x in documents if x]

        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        random.shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)

        documents = None
    
    def create_training_instance(self, index):

        instance = []
        # Need to add [bos] + [eos] tokens
        max_num_tokens = self.max_seq_length - 2

        document = copy.deepcopy(self.documents[index])
        x = truncate_input_sequence(document[0], max_num_tokens)
        y = truncate_input_sequence(document[1], max_num_tokens)

        instance.append(TokenInstance(x, y, len(y[0])))

        return instance
    
    def merge_documents(self, instance_lists):

        self.instances = sum(instance_lists, [])
        self.len = len(self.instances)

def process_data(begin, end):
    print('Start Processs:',begin,'->',end)
    path = './raw_data_tri'
    input_files = os.listdir(path)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    instance_lists = []
    for i in range(begin, min(end,len(input_files))):
        d_c = BioBARTPretrainDataCreator(
            path = os.path.join(path, input_files[i]),
            tokenizer = tokenizer,
            max_seq_length = 302,
            readin = 2000000,
            dupe_factor = 1,
            sentence_permute_prob = 1,
        )
        instance_lists.append(d_c.instances)
    d_c.merge_documents(instance_lists)
    d_c.save('./tokenized_data/genrated_'+str(begin//10).rjust(3,'0')+'.pkl')


if __name__ == '__main__':
    # process_data(0,10)
    threads = []
    path = './raw_data_tri'
    save_path = "./tokenized_data"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    input_files = os.listdir(path)
    for idx in range(0, len(input_files), 10):
        if len(threads) == 100:
            for t in threads:
                t.join()
            threads = []
        else:
            thread = Process(target=process_data, args=(idx, idx+10,))
            thread.start()
            threads.append(thread)
    
    for t in threads:
        t.join()

            