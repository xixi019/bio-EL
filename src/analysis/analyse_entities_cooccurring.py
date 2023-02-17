from kbguided_pretrain.datagen.generate_raw_ptdata import UMLS


def main():
    umls = UMLS("/data1/el/2017AA-active/2017AA/META", only_load_dict=True)


if __name__ == '__main__':
    main()
