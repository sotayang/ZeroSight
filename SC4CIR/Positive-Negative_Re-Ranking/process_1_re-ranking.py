import clip
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import json
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_results_path", type=str, help="Folder path of your model results with similarities", required=True)
parser.add_argument("--process_1_model_results_path", type=str, help="Folder path of your model results of process 1", required=True)
parser.add_argument("--output_process_1_rerank_path", type=str, help="Folder path saving results after process 1 re-ranking", required=True)

args = parser.parse_args()

with open(args.model_results_path, 'r', encoding='utf-8') as file:
    results = json.load(file)

with open(args.process_1_model_results_path, 'r', encoding='utf-8') as file:
    datas = json.load(file)

rank_set = {}

for data in datas:
    if str(data["ori_id"]) in rank_set:
        rank_set[str(data["ori_id"])]["candidate_img_id"].append(data["reference_img_id"])
        rank_set[str(data["ori_id"])]["sim_values"].append(data["similarity"])
    else:
        rank_set[str(data["ori_id"])] = {
            "ori_reference_img_id":data["ori_reference_img_id"],
            "candidate_img_id":[data["reference_img_id"]],
            "sim_values":[data["similarity"]]
        }

output_data = {}
for key, value in rank_set.items():

    ##############################################################################################
    ##############################################################################################
    ###############                                                                ###############
    ############### Score of process 1 = Original similarity + value["sim_values"] ###############
    ###############                                                                ###############     
    ##############################################################################################
    ##############################################################################################

    zipped_pairs = zip(str(value["candidate_img_id"]), Score_of_process_1)
    sorted_pairs = sorted(zipped_pairs, key=lambda x: x[1])

    Score_of_process_1 = [element[1] for element in sorted_pairs]
    candidate_target_imgs = [element[0] for element in sorted_pairs]

    output_data[str(key)] = candidate_target_imgs

with open(args.output_process_1_rerank_path, "w") as f:
    f.write(json.dumps(output_data, indent=4))
