# -*- coding: utf-8 -*-
import os, sys
from glob import glob
import json
import math
import random

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--targetImgs_files_path", type=str, help="Folder path of files containing target images after self ViT filtering", required=True)
parser.add_argument("--gpt4o_relativeCaptions_files_path", type=str, help="Folder path of files containing candidate relative captions after GPT4o generation", required=True)

args = parser.parse_args()

imagesAll = []

prompt = """
        Task Description:
        You will be given two images in sequence. You need to generate a declarative sentence that, when combined with the content of the first image, will enable a search engine to accurately retrieve the second image.

        Output Example:
        "should add more people in a bright room."

        Requirement:
        1. The output sentence should describe the modifications needed to change the first image into the second image.
        2. The subject of the output sentence should be "the first image," and the predicate can be a series of verbs such as "increase", "enlarge", "reduce", "show", "zoom", etc. However, the output must follow the given output example, and the subject must be omitted.
        3. Only this sentence can be output, and no other characters are allowed.
        4. The declarative sentence must be output in English.
        5. The modifications described in the declarative sentence should focus more on the elements within the images (people, objects, colors, numbers, environments, etc.).

        Input Image Sequence:

        """

for root, dirs, files in os.walk(args.targetImgs_files_path):
    files.sort()
    for name in files:
        output_first = []
        referImgPath = os.path.join(root,name)
        imagesName, _ = os.path.splitext(name)
        with open(referImgPath, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        if len(datas) > 0:
            for data in datas:
                output_second = []
                referImg = data['referenceImg']
                similarImgs = data['smiImgs']
                if len(similarImgs) < 2:
                    continue
                for smiImg in similarImgs:
                    imagesFetched = [referImg, smiImg]
                    gpt4o_client = GPT()
                    answer = gpt4o_client.vision(prompt, imagesFetched, max_cycle = 20)
                    while answer is None:
                        answer = gpt4o_client.vision(prompt, imagesFetched, max_cycle = 20)
                    output_data = {
                        'referenceImg':referImg,
                        'smilarImg':smiImg,
                        'relativeCaption':answer
                    }

                    output_second.append(output_data)

                out_item = {
                    'pairs':output_second
                }

                output_first.append(out_item)
        json_filename = args.gpt4o_relativeCaptions_files_path + '/' + imagesName + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output_first, json_file, ensure_ascii=False, indent=4)
        print(f"Data has been written to {json_filename}")