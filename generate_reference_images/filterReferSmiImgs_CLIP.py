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
parser.add_argument("--vit_referImgs_files_path", type=str, help="Folder path of files containing candidate reference images after ViT filtering", required=True)
parser.add_argument("--referImgs_files_path", type=str, help="Folder path of files containing reference images", required=True)

args = parser.parse_args()

for root, dirs, files in os.walk(args.vit_referImgs_files_path):
    for name in tqdm(files):
        referImgPath = os.path.join(root,name)
        imagesName, _ = os.path.splitext(name)
        with open(referImgPath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        referImgs = data['referenceImgs']

        if len(referImgs) > 1:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            model, preprocess = clip.load("ViT-L/14", device=device)
            model.to(device)
            images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in referImgs]
            with torch.no_grad():
                image_features = [model.encode_image(image) for image in images]
            image_features = [feature.cpu().numpy().flatten() for feature in image_features]
            def cosine_similarity(feature1, feature2):
                return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

            lastImgs = []
            lastImgs.extend(referImgs)

            for i in range(len(image_features)-1):
                for j in range(i+1,len(image_features)):
                    similarity = float(cosine_similarity(image_features[i], image_features[j]))
                    if similarity >= 0.8:
                        if referImgs[i] in lastImgs and referImgs[j] in lastImgs:
                            lastImgs.remove(referImgs[j])

            output = {
                'referenceImgs':lastImgs
            }
        else:
            output = {
                'referenceImgs':referImgs
            }

        json_filename = args.referImgs_files_path + imagesName + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)

        print(f"Data has been written to {json_filename}")