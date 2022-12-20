import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

default_umls_filepath = Path()


def prepare_degree_index(umls_filepath: Path):
    degree_index = {}
    return degree_index


def calculate_stats_from_degrees_list(degrees_list: list) -> dict:
    degrees_list = np.array(degrees_list)

    maximum = degrees_list.max(initial=None)
    minimum = degrees_list.min(initial=None)
    mean = degrees_list.mean()
    median = np.median(degrees_list)
    percentiles = list(range(0,101, 10))
    percentile_values = np.percentile(degrees_list, percentiles)
    stats = {
        "maximum": float(maximum),
        "minimum": float(minimum),
        "mean": float(mean),
        "median": float(median)
    }
    for percentile, percentile_value in zip(percentiles, percentile_values):
        stats[f"{percentile}"] = float(percentile_value)
    return stats


def calculate_stats(list_of_entities: list, umls_filepath: Path = default_umls_filepath):
    degree_index = prepare_degree_index(umls_filepath)

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

    print("All mentions")
    print(json.dumps(calculate_stats_from_degrees_list(encountered_degrees), indent=4))
    print("All entities")
    print(json.dumps(calculate_stats_from_degrees_list(encountered_degrees_unique), indent=4))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("filename", type=Path)
    args = argparser.parse_args()

    calculate_stats(json.load(args.filename))


