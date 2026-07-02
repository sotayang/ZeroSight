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
parser.add_argument("--anti_caption_path", type=str, help="Folder path saving anti-caption of your model results", required=True)
parser.add_argument("--output_rerank_path", type=str, help="Folder path saving results after process 1 and 2 re-ranking", required=True)

args = parser.parse_args()

with open(args.model_results_path, 'r', encoding='utf-8') as file:
    results = json.load(file)

with open(args.process_1_model_results_path, 'r', encoding='utf-8') as file:
    datas_process_1 = json.load(file)

with open(args.anti_caption_path, 'r', encoding='utf-8') as file:
    datas_process_2 = json.load(file)


rank_set = {}
for data in datas_process_1:
    if str(data["ori_id"]) in rank_set:
        rank_set[str(data["ori_id"])]["candidate_img_id"].append(data["reference_img_id"])
        rank_set[str(data["ori_id"])]["sim_values"].append(data["similarity"])
    else:
        rank_set[str(data["ori_id"])] = {
            "ori_reference_img_id":data["ori_reference_img_id"],
            "candidate_img_id":[data["reference_img_id"]],
            "sim_values":[data["similarity"]]
        }

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-L/14", device=device)
model.to(device)

assert next(model.parameters()).is_cuda, "Model is not on GPU."

output_data = {}
for data in datas_process_2:
    relative_caption = data["relative_caption"]
    anti_captions = data["anti_captions"]

    text_inputs = torch.cat([clip.tokenize(text) for text in [relative_caption] + anti_captions]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features @ text_features.T

    similarities_process_2 = []
    for i, score in enumerate(similarity[0, 1:]):
        similarities_process_2.append(float(score.item()))
    output_candidate_Imgs_Id_process_2 = data["candidate_Imgs_Id"]

    similarities_process_1 = rank_set[str(data["id"])]["sim_values"]
    output_candidate_Imgs_Id_process_1 = rank_set[str(data["id"])]["candidate_img_id"]

    ##########################################################################################################################################
    ##########################################################################################################################################
    ###############                                                                                                            ###############
    ###############    Score after process 1 and 2 = Original similarity + similarities_process_1 + similarities_process_2     ###############
    ###############                                                                                                            ###############     
    ##########################################################################################################################################
    ##########################################################################################################################################

    zipped_pairs = zip(output_candidate_Imgs_Id, Score_after_process_1_and_2)
    sorted_pairs = sorted(zipped_pairs, key=lambda x: x[1])

    Score_after_process_1_and_2 = [element[1] for element in sorted_pairs]
    output_candidate_Imgs_Id = [element[0] for element in sorted_pairs]

    output_data[str(data["id"])] = output_candidate_Imgs_Id

with open(args.output_rerank_path, "w") as f:
    f.write(json.dumps(output_data, indent=4))