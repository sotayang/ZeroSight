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
parser.add_argument("--video_frame_folders_path", type=str, help="Folder path of folders containing video frames", required=True)
parser.add_argument("--referImgs_files_path", type=str, help="Folder path of files containing reference images", required=True)
parser.add_argument("--vit_targetImgs_files_path", type=str, help="Folder path of files containing candidate target images after ViT filtering", required=True)

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

for root, dirs, files in os.walk(args.referImgs_folders_path):
    files.sort()
    for name in tqdm(files):
        output = []
        referImgPath = os.path.join(root,name)
        with open(referImgPath, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        referImgs = datas['referenceImgs']
        imagesName, _ = os.path.splitext(name)
        imagesPath = args.video_frame_folders_path + imagesName

        for referImg in referImgs:
            imagesAll = []
            for allRoot, allDirs, allFiles in os.walk(imagesPath):
                allFiles.sort()
                for image in allFiles:
                    if  os.path.join(allRoot,image) != referImg:
                        imagesAll.append(os.path.join(allRoot,image))

            image_paths = [referImg]
            image_paths.extend(imagesAll)
            images = [preprocess_image(image_path).to(device) for image_path in image_paths]

            with torch.no_grad():
                image_features = [model(**image) for image in images]

            last_features = [image_feature.last_hidden_state[:, 0, :] for image_feature in image_features]

            referFeature = last_features[0]
            allFeatures = last_features[1:]

            similarities = [float(F.cosine_similarity(referFeature, f).item()) for f in allFeatures]

            smiImgs = []
            smiVal = []

            for i in range(len(similarities)):
                if similarities[i] >= 0.35 and similarities[i] <= 0.50:
                    smiImgs.append(imagesAll[i])
                    smiVal.append(similarities[i])

            output_data = {
                "referenceImg":referImg,
                "smiImgs":smiImgs,
                "similarities":smiVal
            }

            output.append(output_data)

        json_filename = args.vit_targetImgs_folders_path + imagesName + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)
        print(f"Data has been written to {json_filename}")