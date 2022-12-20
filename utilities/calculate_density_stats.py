import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm


def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
        # for _ in range(100000):
            yield line
            line = f.readline()
    return


def prepare_degree_index(umls_rel_filename: Path):
    rel_reader = byLineReader(umls_rel_filename)

    cui2triple = defaultdict(set)
    for line in tqdm(rel_reader, ascii=True, total=24633828):
        l = line.strip().split("|")
        if len(l[7]) > 0:
            cui = l[0]
            rel = l[7]
            object = l[4]

            cui2triple[cui].add((rel, object))
    degree_index = {key: len(value) for key, value in cui2triple.items()}
    return degree_index


def calculate_stats_from_degrees_list(degrees_list: list) -> dict:
    degrees_list_array = np.array(degrees_list)

    maximum = degrees_list_array.max(initial=None)
    minimum = degrees_list_array.min(initial=None)
    mean = degrees_list_array.mean()
    median = np.median(degrees_list_array)
    percentiles = list(range(0,101, 10))
    percentile_values = np.percentile(degrees_list_array, percentiles)
    stats = {
        "maximum": float(maximum),
        "minimum": float(minimum),
        "mean": float(mean),
        "median": float(median),
        "counter": len(degrees_list)
    }
    for percentile, percentile_value in zip(percentiles, percentile_values):
        stats[f"{percentile}"] = float(percentile_value)
    return stats


def calculate_stats(list_of_entities: list, degree_index: dict):
    num_entities_not_found = 0
    encountered_degrees_unique = []
    encountered_degrees = []
    for entity_identifier in list_of_entities:
        if entity_identifier in degree_index:
            encountered_degrees.append(degree_index[entity_identifier])
    for entity_identifier in set(list_of_entities):
        if entity_identifier not in degree_index:
            num_entities_not_found += 0
        else:
            encountered_degrees_unique.append(degree_index[entity_identifier])

    assert len(encountered_degrees) > 0
    print("All mentions")
    print(json.dumps(calculate_stats_from_degrees_list(encountered_degrees), indent=4))
    print("All entities")
    print(json.dumps(calculate_stats_from_degrees_list(encountered_degrees_unique), indent=4))
    print(f"Entities not found: {num_entities_not_found}")


def main(filename: Path, umls_rel_filename: Path):
    degree_index = prepare_degree_index(umls_rel_filename)
    calculate_stats(json.load(filename.open())["id"], degree_index)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("filename", type=Path)
    argparser.add_argument("--umls_rel_filename", type=Path, default="./MRREL.RRF")
    args = argparser.parse_args()

    main(args.filename, args.umls_rel_filename)



