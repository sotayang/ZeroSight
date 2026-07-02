import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from torchvision import transforms
import torch.nn.functional as F
import os
import json
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--clip_targetImgs_files_path", type=str, help="Folder path of files containing candidate target images after CLIP filtering", required=True)
parser.add_argument("--targetImgs_files_path", type=str, help="Folder path of files containing target images after self ViT filtering", required=True)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')
    print("GPU is not available, using CPU")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k').to(device)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

for root, dirs, files in os.walk(args.clip_targetImgs_files_path):
    for name in tqdm(files):
        referImgPath = os.path.join(root,name)
        imagesName, _ = os.path.splitext(name)
        output = []
        with open(referImgPath, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        for data in datas:
            referenceImg = data['referenceImg']
            allImgs = data['allImgs']
            
            images = [preprocess_image(allImg).to(device) for allImg in allImgs]

            with torch.no_grad():
                image_features = [model(**image) for image in images]

            last_features = [image_feature.last_hidden_state[:, 0, :] for image_feature in image_features]

            lastImgs = []
            lastImgs.extend(allImgs)

            allsmi = []

            for i in range(len(last_features)-1):
                temp = []
                for j in range(i+1,len(last_features)):
                    similarity = float(F.cosine_similarity(last_features[i], last_features[j]).item())
                    if similarity >= 0.85:
                        if allImgs[i] in lastImgs and allImgs[j] in lastImgs:
                            lastImgs.remove(allImgs[j])
                        continue
                    temp.append(similarity)
                allsmi.append(temp)

            output_data = {
                'referenceImg':referenceImg,
                'smiImgs':lastImgs,
                'allsmi':allsmi
            }

            output.append(output_data)

        json_filename = args.targetImgs_files_path + imagesName + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)
        print(f"Data has been written to {json_filename}")