import clip
import torch
from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm

from torchvision import transforms

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--model_results_path", type=str, help="Folder path of your model results with similarities", required=True)
parser.add_argument("--anti_caption_path", type=str, help="Folder path saving anti-caption of your model results", required=True)
parser.add_argument("--output_process_2_rerank_path", type=str, help="Folder path saving results after process 2 re-ranking", required=True)

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-L/14", device=device)
model.to(device)

assert next(model.parameters()).is_cuda, "Model is not on GPU."


with open(args.model_results_path, 'r', encoding='utf-8') as file:
    results = json.load(file)

anti_caption_path = args.anti_caption_path
with open(anti_caption_path, 'r', encoding='utf-8') as file:
    datas = json.load(file)

output_data = {}
for data in datas:
    relative_caption = data["relative_caption"]
    anti_captions = data["anti_captions"]

    text_inputs = torch.cat([clip.tokenize(text) for text in [relative_caption] + anti_captions]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = text_features @ text_features.T

    similarities = []
    for i, score in enumerate(similarity[0, 1:]):
        similarities.append(float(score.item()))

    output_candidate_Imgs_Id = data["candidate_Imgs_Id"]

    ##############################################################################################
    ##############################################################################################
    ###############                                                                ###############
    ###############    Score of process 2 = Original similarity + similarities     ###############
    ###############                                                                ###############     
    ##############################################################################################
    ##############################################################################################

    zipped_pairs = zip(output_candidate_Imgs_Id, Score_of_process_2)
    sorted_pairs = sorted(zipped_pairs, key=lambda x: x[1])

    Score_of_process_2 = [element[1] for element in sorted_pairs]
    output_candidate_Imgs_Id = [element[0] for element in sorted_pairs]

    output_data[str(data["id"])] = output_candidate_Imgs_Id

with open(args.output_process_2_rerank_path, "w") as f:
    f.write(json.dumps(output_data, indent=4))

