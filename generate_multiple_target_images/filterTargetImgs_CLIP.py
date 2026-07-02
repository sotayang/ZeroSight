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
parser.add_argument("--vit_targetImgs_files_path", type=str, help="Folder path of files containing candidate target images after ViT filtering", required=True)
parser.add_argument("--clip_targetImgs_files_path", type=str, help="Folder path of files containing candidate target images after CLIP filtering", required=True)

args = parser.parse_args()

for root, dirs, files in os.walk(args.vit_targetImgs_files_path):
    for name in tqdm(files):
        imagesAll = []
        referImgPath = os.path.join(root,name)
        imagesName, _ = os.path.splitext(name)
        output = []
        with open(referImgPath, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        if len(datas) > 0:
            for data in datas:
                referImg = data['referenceImg']
                smiImgs = data['smiImgs']
                similarities_ViT = data['similarities']

                if len(smiImgs) == 0:
                    continue
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                model, preprocess = clip.load("ViT-L/14", device=device)
                model.to(device)
                image_paths = [referImg]
                image_paths.extend(smiImgs)
                images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]
                with torch.no_grad():
                    image_features = [model.encode_image(image) for image in images]
                image_features = [feature.cpu().numpy().flatten() for feature in image_features]

                def cosine_similarity(feature1, feature2):
                    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

                referFeature = image_features[0]
                allFeatures = image_features[1:]

                similarities = [float(cosine_similarity(referFeature, f)) for f in allFeatures]

                smiImgsFiltered = []
                smiVal_ViT = []
                smiVal_CLIP = []

                for i in range(len(similarities)):
                    if similarities[i] >= 0.65 and similarities[i] <= 0.75:
                        smiImgsFiltered.append(smiImgs[i])
                        smiVal_ViT.append(similarities_ViT[i])
                        smiVal_CLIP.append(similarities[i])

                if len(smiImgsFiltered) == 0:
                    continue

                output_data = {
                    "referenceImg":referImg,
                    "allImgs":smiImgsFiltered,
                    "similarities_ViT":smiVal_ViT,
                    "similarities_CLIP":smiVal_CLIP
                }

                output.append(output_data)

        json_filename = args.clip_targetImgs_files_path + imagesName + '.json'

        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)

        print(f"Data has been written to {json_filename}")
