import os
import ipdb
import csv
import sys
csv.field_size_limit(sys.maxsize)
import json

dir = './'
# main function to read files, gather entity and dump the dictionary into json file
def read_data(dir=dir, task = 'ncbi'):
    """
    dir : where you save the finetuning files
    task: ncbi, CDR_Data, cometa or AskAPatient
    """
    task_dir = dir + task + '/'
    # get the files from different dataset
    files = get_target_file(task_dir, task)
    # 
    ents = get_target_ent(files, task)
    print(ents)
    with open(task+'finetune_data.json', 'w') as f:
        json.dump(ents, f)
        print('entity id and type information is saved.')


def get_target_file(task_dir, task):
    if task == 'ncbi':
        files = [task_dir+file for file in os.listdir(task_dir) if file.endswith('txt')]
    if task == 'CDR_Data':
        file = []
        pass
    if task == 'cometa':
        files = [task_dir+'COMETA_id_sf_dictionary.txt']
    if task == 'AskAPatient':
        files = [task_dir+file for file in os.listdir(task_dir)]
    return files

def get_target_ent(files, task):
    ent_dic = {"type":[], "id":[]}
    if task == 'ncbi':
        for file in files:
            with open(file, mode = 'r') as f:
                for line in f:
                    if len(line.split('\t')) >=6:
                        type = line.split('\t')[4].strip('\n')
                        id = line.split('\t')[5].strip('\n')
                        ent_dic['type'].append(type)
                        ent_dic['id'].append(id)
    if task == 'cometa':
        with open(files[0]) as file:
            for row in file:
                try:
                    ent_dic['id'].append(row.split('||')[0])
                except IndexError:
                    pass
    if task == "AskAPatient":
        for file in files:
            with open(file, mode = 'r', encoding= 'unicode_escape') as f:
                for line in f:
                    if len(line.split('\t')) ==3:
                        id = line.split('\t')[0]
                        ent_dic['id'].append(id)

        pass
    ent_dic['type'] = list(set(ent_dic['type']))
    ent_dic['id'] = list(set(ent_dic['id']))
    return ent_dic

read_data(dir, 'cometa')