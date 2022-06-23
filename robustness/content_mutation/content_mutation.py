import argparse
import sys
import os
from yaml import load, dump
from typing import List
import pandas as pd
import time
import json
from collections import Counter
import logging
from logging.config import fileConfig
from pathlib import Path


from obfuscation import *
from utils import *


fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def perform_mutations(df, tracker_dict, output_fname, map_dir, change_third_partiness, graph_columns):

	df_graph_new = pd.DataFrame()
	mapping_dict = {}

	try:
		visit_id = df.iloc[0]['visit_id']
		if visit_id in tracker_dict:
			tracker_list = tracker_dict[visit_id]

			df_trackers = df[df['name'].isin(tracker_list)]
			df_others = df[~df['name'].isin(tracker_list)]
			df_copy = df_trackers.copy()

			df_third_party = find_third_parties(df_copy)
			node_names = df_third_party['name'].tolist()

			if change_third_partiness:
				party_value = 1
				top_level_url = df_copy['top_level_url'].iloc[0]
				for nn in node_names:
					obfs_name = obfuscate(nn, party_value, top_level_url)
					mapping_dict[nn] = obfs_name
			else:
				party_value = 3
				for nn in node_names:
					obfs_name = obfuscate(nn, party_value)
					mapping_dict[nn] = obfs_name

			map_fname = os.path.join(map_dir, str(visit_id) + ".json")
			with open(map_fname, 'w') as f:
				f.write(json.dumps(mapping_dict, indent=4))

			#Replace URLs with mutated ones in the graph
			df_graph_new = df_copy.append(df_others)
			df_graph_new = df_graph_new.replace({'name' : mapping_dict, 'src' : mapping_dict, 'dst' : mapping_dict})
			df_graph_new = df_graph_new.drop(['is_third_party'], axis=1)

			if not output_fname.is_file():
				df_graph_new.reindex(columns=graph_columns).to_csv(str(output_fname))
			else:
				df_graph_new.reindex(columns=graph_columns).to_csv(str(output_fname), mode='a', header=False)

			logger.info("Done processing visit_id " + str(visit_id))
		else:
			logger.info("VID " + str(visit_id) + " not in prediction file, so nothing to change.")

	except Exception as e:
		logger.error("Error in content mutation:", e)

def pipeline(graph_fname, output_fname, predictions, map_dir, change_third_partiness):

	all_start = time.time()

	if not os.path.exists(map_dir):
		os.mkdir(map_dir)

	df_graph = pd.read_csv(graph_fname)
	graph_columns = df_graph.columns
	logger.info("Read graph file")

	#Find third parties 
	df_graph['is_third_party'] = df_graph.apply(check_third_party, axis=1)
	logger.info("Found third parties")

	tracker_dict = find_tracker_predictions(predictions)
	logger.info("Obtained all URLs predicted as trackers")

	df_graph.groupby(['visit_id', 'top_level_domain']).apply(perform_mutations, tracker_dict, output_fname, map_dir, 
		change_third_partiness, graph_columns)
	logger.info("Finished mutation for all websites")

	all_end = time.time()
	logger.info("Time for all visits: " + str(all_end - all_start))


def main(program: str, args: List[str]):
    
    parser = argparse.ArgumentParser(prog=program, description="Run the WebGraph classification pipeline.")
    
    parser.add_argument(
        "--graph",
        type=str,
        help="Graph CSV file.",
        default="graph.csv"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output file.",
        default=Path("out.csv")
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Classifier predictions.",
        default="predictions"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        help="Output directory of domain mappings.",
        default="mappings"
    )
    parser.add_argument(
        "--changetp",
        type=bool,
        help="Make third-parties first-party sub-domains.",
        default=False
    )
   
    ns = parser.parse_args(args)
    pipeline(ns.graph, ns.out, ns.predictions, ns.mapping, ns.changetp)


if __name__ == "__main__":

    main(sys.argv[0], sys.argv[1:])
	
